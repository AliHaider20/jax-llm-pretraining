import jax
import flax.nnx as nnx
from orbax import checkpoint
import jax.numpy as jnp
import pandas as pd
from data_loader import tokenizer, maxlen

from jax.sharding import SingleDeviceSharding

from model import MiniGPT 


def generate_text(model, start_tokens, max_new_tokens=50, temperature=1.0):
    tokens = list(start_tokens)

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen:]

        # RIGHT-pad to match training (not left-pad!)
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :] / temperature

        next_token = int(jnp.argmax(next_token_logits))

        if next_token == tokenizer.encode('<|endoflabel|>', allowed_special={'<|endoflabel|>'})[0]:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)

model = MiniGPT(
    vocab_size=tokenizer.n_vocab,
    maxlen=maxlen,
    embed_dim=192,
    num_heads=6,
    feed_forward_dim=int(2/3 * 4 * 192),
    num_transformer_blocks=8,
    rngs=nnx.Rngs(0), 
)


def generate_story(model, story_prompt, temperature, max_new_tokens):
    start_tokens = tokenizer.encode(story_prompt)[:maxlen]
    generated = generate_text(model, start_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
    return generated

cpu_device = jax.devices('cpu')[0]
cpu_sharding = SingleDeviceSharding(cpu_device)
restore_args = jax.tree_util.tree_map(
    lambda _: checkpoint.ArrayRestoreArgs(sharding=cpu_sharding),
    nnx.state(model)
)

def detect_red_team_prompt(story_prompt, temperature, max_new_tokens):
    return generate_story(model, story_prompt, temperature, max_new_tokens)

test_prompts = pd.read_csv("test_data.csv")
print("Running inference on first 5 test prompts")

for _, row in test_prompts.head(5).iterrows():
    user_prompt = row['text'] + ":"
    output = detect_red_team_prompt(user_prompt, 0.2, 30)
    print("Model output:", output)
    print("Ground truth category:", row['category'])
    print("===="*25)