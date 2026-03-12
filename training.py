import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint
import tiktoken
import yaml
import wandb
import matplotlib.pyplot as plt
from jaxlib.xla_client import SingleDeviceSharding
from tqdm import tqdm

from model import MiniGPT
from data_loader import load_and_preprocess_data

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MAX_LENGTH             = config["MAX_LENGTH"]
BATCH_SIZE             = config["BATCH_SIZE"]
NUM_EPOCHS             = config["NUM_EPOCHS"]
NUM_LAYERS             = config["NUM_LAYERS"]
EMBED_DIM              = config["EMBED_DIM"]
NUM_HEADS              = config["NUM_HEADS"]
FEED_FORWARD_DIM       = config["FEED_FORWARD_DIM"]
TOKENIZER_PATH         = config.get("custom_tokenizer_model_path", None)
CHECKPOINT_PATH        = config.get("MODEL_CHECKPOINT_PATH", "checkpoints/minigpt")
SEED                   = 42

# ── Tokenizer ─────────────────────────────────────────────────────────────────
def load_custom_encoding(path: str) -> tiktoken.Encoding:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return tiktoken.Encoding(
        name=data["name"],
        pat_str=data["pat_str"],
        mergeable_ranks=data["mergeable_ranks"],
        special_tokens=data["special_tokens"],
    )

tokenizer = (
    load_custom_encoding(TOKENIZER_PATH)
    if TOKENIZER_PATH
    else tiktoken.get_encoding("o200k_base")
)

vocab_size = tokenizer.n_vocab

# Special token IDs — used for masking
PAD_TOKEN_ID        = 0   # ← replace with your real pad token if different
LABEL_START_TOKEN   = tokenizer.encode("<|startoflabel|>", allowed_special="all")[0]
LABEL_END_TOKEN     = tokenizer.encode("<|endoflabel|>",   allowed_special="all")[0]

# ── Data ──────────────────────────────────────────────────────────────────────
text_dl, batches_per_epoch = load_and_preprocess_data(
    batch_size=BATCH_SIZE, maxlen=MAX_LENGTH, shuffle=True, seed=SEED
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = MiniGPT(
    maxlen=MAX_LENGTH,
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    feed_forward_dim=FEED_FORWARD_DIM,
    num_transformer_blocks=NUM_LAYERS,
    rngs=nnx.Rngs(SEED),
)

# ── Restore checkpoint (if exists) ────────────────────────────────────────────
checkpoint_path = Path.cwd() / CHECKPOINT_PATH          # defined once, used everywhere

if checkpoint_path.exists():
    print(f"Restoring model from {checkpoint_path} ...")
    cpu_device   = jax.devices("cpu")[0]
    cpu_sharding = SingleDeviceSharding(cpu_device)

    restore_args = jax.tree_util.tree_map(
        lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=cpu_sharding),
        nnx.state(model),
    )

    checkpointer   = orbax.checkpoint.PyTreeCheckpointer()
    restored_state = checkpointer.restore(
        checkpoint_path,
        item=nnx.state(model),
        restore_args=restore_args,
    )
    nnx.update(model, restored_state)
    print("Checkpoint restored ✅")

# ── LR schedule & optimiser ───────────────────────────────────────────────────
total_steps   = batches_per_epoch * NUM_EPOCHS
warmup_steps  = max(1, total_steps // 10)

print(f"Total steps : {total_steps:,}")
print(f"Warmup steps: {warmup_steps:,}")

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
    end_value=1e-5,
)

optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate=lr_schedule, weight_decay=0.01),
    wrt=nnx.Param,
)

metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

# ── Loss function (JAX-compatible, masked) ────────────────────────────────────
def loss_fn(model, batch):
    """
    Computes cross-entropy loss **only** over label tokens
    (tokens that appear after <|startoflabel|>), ignoring pad positions.

    Args:
        model : MiniGPT instance
        batch : (inputs, targets)  —  both int32 [B, T]

    Returns:
        loss   : scalar mean loss over unmasked positions
        logits : float32 [B, T, vocab_size]
    """
    inputs, targets = batch                                # [B, T]
    logits = model(inputs)                                 # [B, T, V]

    # 1. Build label-region mask: positions AFTER <|startoflabel|>
    #    cumsum trick works purely with jnp — no Python control flow
    is_label_start = (inputs == LABEL_START_TOKEN).astype(jnp.int32)   # [B, T]
    label_mask     = jnp.cumsum(is_label_start, axis=-1) > 0           # [B, T]

    # 2. Also mask padding tokens in targets
    pad_mask       = (targets != PAD_TOKEN_ID)                          # [B, T]

    # 3. Combined mask
    mask           = label_mask & pad_mask                              # [B, T]

    # 4. Per-token loss
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, targets
    )                                                                   # [B, T]

    # 5. Masked mean  (add small eps to avoid divide-by-zero)
    loss = (per_token_loss * mask).sum() / (mask.sum() + 1e-9)

    return loss, logits


# ── Train step (JIT-compiled) ─────────────────────────────────────────────────
@nnx.jit
def train_step(model, optimizer, metrics, batch):
    """
    Single gradient-update step.

    Args:
        model     : MiniGPT
        optimizer : nnx.Optimizer
        metrics   : nnx.MultiMetric
        batch     : (inputs, targets)  int32 [B, T]

    Returns:
        loss : scalar (for W&B logging outside jit)
    """
    grad_fn          = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)

    metrics.update(loss=loss)          # ← only pass what Average("loss") needs
    optimizer.update(model, grads)

    return loss


# ── Target builder: shift right, pad last position with PAD_TOKEN_ID ──────────
prep_target_batch = jax.vmap(
    lambda tokens: jnp.concatenate(
        (tokens[1:], jnp.array([PAD_TOKEN_ID], dtype=jnp.int32))
    )
)

# ── W&B init ──────────────────────────────────────────────────────────────────
wandb.init(
    project="minigpt-pretraining",
    config={
        "max_length"      : MAX_LENGTH,
        "batch_size"      : BATCH_SIZE,
        "num_epochs"      : NUM_EPOCHS,
        "num_layers"      : NUM_LAYERS,
        "embed_dim"       : EMBED_DIM,
        "num_heads"       : NUM_HEADS,
        "feed_forward_dim": FEED_FORWARD_DIM,
        "vocab_size"      : vocab_size,
        "total_steps"     : total_steps,
        "warmup_steps"    : warmup_steps,
        "peak_lr"         : 3e-4,
        "weight_decay"    : 0.01,
    },
)

# ── Training loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    metrics_history = {"train_loss": []}
    global_step     = 0

    for epoch in range(NUM_EPOCHS):

        epoch_bar = tqdm(
            text_dl,
            total=batches_per_epoch,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in epoch_bar:
            # ── Prepare inputs / targets ──────────────────────────────────────
            input_batch  = jnp.array(batch).T.astype(jnp.int32)           # [B, T]
            target_batch = prep_target_batch(input_batch)                  # [B, T]

            # ── Forward + backward ────────────────────────────────────────────
            loss = train_step(
                model, optimizer, metrics, (input_batch, target_batch)
            )

            global_step += 1

            # ── Log every 2 steps ─────────────────────────────────────────────
            if global_step % 2 == 0:
                computed   = metrics.compute()
                train_loss = float(computed["loss"])
                current_lr = float(lr_schedule(global_step))

                metrics_history["train_loss"].append(train_loss)
                metrics.reset()

                # tqdm postfix
                epoch_bar.set_postfix(
                    loss=f"{train_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                )

                # W&B log
                wandb.log(
                    {
                        "train/loss"       : train_loss,
                        "train/lr"         : current_lr,
                        "train/epoch"      : epoch + 1,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer.save(str(checkpoint_path), nnx.state(model), force=True)
    print(f"Model saved → {checkpoint_path}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(metrics_history["train_loss"])
    plt.title("Training Loss")
    plt.xlabel("Step (×2)")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("train_loss.png", dpi=150)
    wandb.log({"train/loss_curve": wandb.Image("train_loss.png")})
    plt.show()

    wandb.finish()