import pickle
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tiktoken
import optax
from model import MiniGPT
from data_loader import load_and_preprocess_data
from pathlib import Path
import orbax
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MAX_LENGTH = config["MAX_LENGTH"]
SEED = 42
BATCH_SIZE = config["BATCH_SIZE"]
num_epochs = config["NUM_EPOCHS"]
num_transformer_blocks = config["NUM_LAYERS"]
embed_dim = config["EMBED_DIM"]
num_heads = config["NUM_HEADS"]
feed_forward_dim = config["FEED_FORWARD_DIM"]
tokenizer_path = config['custom_tokenizer_model_path']


def load_custom_encoding(path: str) -> tiktoken.Encoding:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return tiktoken.Encoding(
        name=data["name"],
        pat_str=data["pat_str"],
        mergeable_ranks=data["mergeable_ranks"],
        special_tokens=data["special_tokens"],
    )

if tokenizer_path:
    tokenizer = load_custom_encoding(tokenizer_path) # Loading a custom tokenizer
else:
    tokenizer = tiktoken.get_encoding("o200k_base")

vocab_size = tokenizer.n_vocab

text_dl, batches_per_epoch = load_and_preprocess_data(
    batch_size=BATCH_SIZE, maxlen=MAX_LENGTH, shuffle=False, seed=SEED
)

model = MiniGPT(
    maxlen=MAX_LENGTH,
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    feed_forward_dim=feed_forward_dim,
    num_transformer_blocks=num_transformer_blocks,
    rngs=nnx.Rngs(0),
)

total_steps = batches_per_epoch * num_epochs
warmup_steps = max(1, total_steps // 10)  # 10% warmup

print(f"Total training steps: {total_steps:,}")
print(f"Warmup steps: {warmup_steps:,}")

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
    end_value=1e-5,
)

optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate=lr_schedule, weight_decay=0.01), wrt=nnx.Param
)

metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model, batch):
    """
    This function is used to calculate loss between predicted and ground truth tokens

    Args:
        model: Trained model
        batch: Batch for predicted and ground truth tokens

    Returns:
        loss: Calculated loss between predicted and ground truth tokens
        logits: Logits predicted by the model for the input batch
    """

    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, logits


if __name__ == "__main__":

    @nnx.jit
    def train_step(model, optimizer, metrics, batch):
        """


        Args:
            model: model to be trained
            optimizer: optimizer for the training step
            metrics: metrics to be updated with the loss and logits for the batch
            batch: batch for the step
        """
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)

        metrics.update(loss=loss, logits=logits, labels=batch[1])
        optimizer.update(model, grads)

    metrics_history = {"train_loss": []}

    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    for epoch in range(num_epochs):
        step = 0
        for batch in text_dl:
            input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
            target_batch = prep_target_batch(jnp.array(jnp.array(batch).T)).astype(
                jnp.int32
            )
            print(".", end="")
            train_step(model, optimizer, metrics, (input_batch, target_batch))

            if (step + 1) % 2 == 0:
                for metric, value in metrics.compute().items():
                    metrics_history[f"train_{metric}"].append(value)
                metrics.reset()

                current_lr = lr_schedule(step)
                print(
                    f"\nEpoch: {epoch + 1}, Step {step + 1}, Loss: {metrics_history['train_loss'][-1]:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
            step += 1

    plt.plot(metrics_history["train_loss"])
    plt.title("Training Loss — 3 steps, 100 stories")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    checkpointer.save(config['MODEL_CHECKPOINT_PATH'], nnx.state(model), force=True)
    print(f"Model saved as {checkpoint_path}")
