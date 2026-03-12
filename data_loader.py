import tiktoken
import grain as pygrain
import jax.numpy as jnp
from dotenv import load_dotenv
import pickle

load_dotenv()

def load_custom_encoding(path: str) -> tiktoken.Encoding:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return tiktoken.Encoding(
        name=data["name"],
        pat_str=data["pat_str"],
        mergeable_ranks=data["mergeable_ranks"],
        special_tokens=data["special_tokens"],
    )

tokenizer = load_custom_encoding("red_teaming_tokenizer.pkl")

class RedTeamingDataset:
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode('<|endoflabel|>', allowed_special="all")[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special="all")

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens

def load_and_preprocess_data(
    batch_size,
    maxlen,
    num_epochs = 1,
    shuffle = False,
    seed = 42
):
    """
    Load and preprocess red teaming prompts data with memory-efficient chunk reading.

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
    with open("train_data.txt", "r", encoding="utf-8") as f:
        raw_data = f.read()
        prompts = raw_data.split("\n")

    print(f"Loaded {len(prompts)} red teaming prompts")
    
    if len(prompts) == 0:
        raise ValueError("No valid datapoints found in the dataset")

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
    
