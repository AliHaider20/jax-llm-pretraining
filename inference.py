import jax
import jax.numpy as jnp
import flax.nnx as nnx
from orbax import checkpoint
import orbax
from pathlib import Path
import pandas as pd
from jax.sharding import SingleDeviceSharding

from data_loader import tokenizer
from model import MiniGPT
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── Pre-encode special tokens once ───────────────────────────────────────────
_encode = lambda t: tokenizer.encode(t, allowed_special="all")[0]

END_OF_PROMPT_TOKEN  = _encode("<|endofprompt|>")
START_OF_LABEL_TOKEN = _encode("<|startoflabel|>")
END_OF_LABEL_TOKEN   = _encode("<|endoflabel|>")


def load_model_from_checkpoint():
    """Load model and restore from checkpoint."""
    model = MiniGPT(
        vocab_size=tokenizer.n_vocab,
        maxlen=config["MAX_LENGTH"],
        embed_dim=config["EMBED_DIM"],
        num_heads=config["NUM_HEADS"],
        feed_forward_dim=config["FEED_FORWARD_DIM"],
        num_transformer_blocks=config["NUM_LAYERS"],
        rngs=nnx.Rngs(0),
    )

    cpu_device = jax.devices("cpu")[0]
    cpu_sharding = SingleDeviceSharding(cpu_device)
    restore_args = jax.tree_util.tree_map(
        lambda _: checkpoint.ArrayRestoreArgs(sharding=cpu_sharding),
        nnx.state(model),
    )

    checkpoint_path = Path.cwd() / "new_model_checkpoint.orbax"
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    restored_state = checkpointer.restore(
        checkpoint_path,
        item=nnx.state(model),
        restore_args=restore_args,
        )
    nnx.update(model, restored_state)
    
    return model


def generate_text(
    model,
    start_tokens: list[int],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    seed: int = 42,
) -> str:
    """
    Auto-regressively generate tokens from `start_tokens`.

    Args:
        model          : MiniGPT instance
        start_tokens   : list of token IDs (already includes prompt + trigger tokens)
        max_new_tokens : maximum tokens to generate
        temperature    : >1 → more random, <1 → sharper/greedier; 0 → greedy
        seed           : JAX PRNG seed for reproducibility

    Returns:
        Decoded string of the full sequence (prompt + generated label)
    """
    # Convert start_tokens to regular Python integers
    tokens = [int(t) for t in start_tokens]
    rng    = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        context    = tokens[-model.maxlen:]
        actual_len = len(context)

        # Right-pad to maxlen (consistent with training)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]          # [1, T]
        logits        = model(context_array)                  # [1, T, V]

        # Logits at the last *real* token position
        next_token_logits = logits[0, actual_len - 1, :]     # [V]

        # ── Sampling ─────────────────────────────────────────────────────────
        if temperature == 0.0:
            # Pure greedy — temperature=0 is a common convention
            next_token = int(jnp.argmax(next_token_logits))
        else:
            # ✅ FIX 3 & 4: actually use temperature + JAX random sampling
            scaled_logits = next_token_logits / temperature
            rng, subkey  = jax.random.split(rng)
            next_token    = int(jax.random.categorical(subkey, scaled_logits))

        if next_token == END_OF_LABEL_TOKEN:
            break

        tokens.append(next_token)

    # Ensure all tokens are Python integers before decoding
    tokens = [int(t) for t in tokens]
    return tokenizer.decode(tokens)


def detect_red_team_prompt(
    model,
    raw_prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 100,
    seed: int = 42,
) -> str:
    """
    Wrap `raw_prompt` in the training format and run generation.

    Training format:
        <|startofprompt|>{text}<|endofprompt|><|startoflabel|>

    Args:
        model          : MiniGPT instance
        raw_prompt     : plain text prompt (no special tokens)
        temperature    : sampling temperature
        max_new_tokens : max label tokens to generate
        seed           : PRNG seed

    Returns:
        Full decoded output (prompt + predicted label)
    """
    formatted_prompt = (
        f"<|startofprompt|>{raw_prompt}<|endofprompt|><|startoflabel|>"
    )

    start_tokens = tokenizer.encode(
        formatted_prompt,
        allowed_special="all",          # allow our custom special tokens
    )[:config["MAX_LENGTH"]]

    return generate_text(
        model,
        start_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )

if __name__ == "__main__":
    # ── Model ─────────────────────────────────────────────────────────────────────
    model = load_model_from_checkpoint()
    print("Model loaded from checkpoint ✅")

    # ── Inference ─────────────────────────────────────────────────────────────────
    test_prompts = pd.read_csv("test_data.csv")
    print("Running inference on first 5 test prompts\n")

    for i, (_, row) in enumerate(test_prompts.sample(5).iterrows()):
        output = detect_red_team_prompt(
            model,
            raw_prompt=row["text"],      # raw text — special tokens added inside
            temperature=0.8,
            max_new_tokens=100,
            seed=i,                      # different seed per sample for variety
        )
        print(f"Model output    : {output}")
        print("-" * 60)
        print(f"Ground truth    : {row['category']}")
        print("=" * 60 + "\n")