from datasets import load_dataset
import tiktoken
import grain as pygrain
import jax.numpy as jnp
from dotenv import load_dotenv
import os

load_dotenv()

tokenizer = tiktoken.get_encoding("gpt2")

vocab_size = tokenizer.n_vocab
num_transformer_blocks = 8
maxlen = 128
embed_dim = 192
num_heads = 6
feed_forward_dim = int(2/3 * 4 * embed_dim)
SEED = 42

tokenizer = tiktoken.get_encoding("gpt2")

class RedTeamingDataset:
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={'<|endoftext|>'})

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens



def load_and_preprocess_data(
    batch_size,
    maxlen,
    max_stories = 100_000,
    num_epochs = 1,
    shuffle = False,
    seed = 42
):
    """
    Load and preprocess TinyStories data with memory-efficient chunk reading.

    Args:
        file_path: Path to the text file
        batch_size: Batch size for training
        maxlen: Maximum sequence length
        max_stories: Maximum number of stories to load (for memory efficiency)
        num_epochs: Number of training epochs
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Grain DataLoader, estimated_batches_per_epoch)
    """
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("aurora-m/redteam", token=os.getenv("HF_TOKEN"), split='train')
    df = ds.to_pandas()[['text', 'category']].dropna(subset=['category'])


    def get_final_text(data):
        data = data['text'].split("\n")[1] + ":" + data['category'] + "<|endoftext|>" + "\n"
        return data

    prompts = df.apply(get_final_text, axis=1).tolist()
    print(f"Loaded {len(prompts)} red teaming prompts")
    
    if len(prompts) == 0:
        raise ValueError("No valid stories found in the dataset")

    # Calculate estimated batches per epoch
    estimated_batches_per_epoch = len(prompts) // batch_size
    print(f"Estimated batches per epoch: {estimated_batches_per_epoch:,}")

    # Create efficient dataset
    dataset = RedTeamingDataset(prompts, maxlen, tokenizer)

    # Configure sampler with sharding support
    sampler = pygrain.samplers.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=pygrain.sharding.NoSharding(),
        num_epochs=num_epochs,
    )

    # Create DataLoader with efficient batching
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            pygrain.transforms.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    print(f"Created DataLoader with batch_size={batch_size}, maxlen={maxlen}")
    return dataloader, estimated_batches_per_epoch
    

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

        if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)


def generate_story(model, story_prompt, temperature, max_new_tokens):
    start_tokens = tokenizer.encode(story_prompt)[:maxlen]
    generated = generate_text(model, start_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
    return generated


