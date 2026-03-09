
# Red Teaming LLM: Pre-training with JAX

A lightweight implementation of MiniGPT, a transformer-based language model pre-trained on the TinyStories dataset using JAX and Flax. This project demonstrates modern deep learning practices including efficient data loading, advanced training techniques, and model checkpointing.

## Overview

This project implements a pre-training pipeline for a small-scale GPT-like model with the following features:

- **Model Architecture**: MiniGPT - a transformer-based language model with:
  - Token and position embeddings
  - Multi-head self-attention
  - Causal masking for autoregressive generation
  - Configurable transformer blocks
  
- **Framework**: [JAX](https://github.com/google/jax) with [Flax NNX](https://github.com/google/flax) for efficient neural network operations
- **Training**: Advanced optimization with:
  - Warmup cosine decay learning rate schedule
  - AdamW optimizer with weight decay
  - Custom tokenizer based on tiktoken
  - Memory-efficient data loading with Grain

- **Inference**: Text generation with temperature-based sampling

## Prerequisites

- Python 3.12+
- pip or uv package manager
- GPU support (optional, but recommended for faster training)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd red_teaming_llm
```

2. Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (faster)
uv sync
```

## Configuration

Edit `config.yaml` to customize model and training parameters:

```yaml
# Model Architecture
NUM_LAYERS: 6              # Number of transformer blocks
EMBED_DIM: 258            # Embedding dimension
NUM_HEADS: 6              # Number of attention heads
FEED_FORWARD_DIM: 688     # Feed-forward layer dimension

# Training
NUM_EPOCHS: 7             # Number of training epochs
BATCH_SIZE: 64            # Batch size
LEARNING_RATE: 1e-4       # Base learning rate
MAX_LENGTH: 128           # Maximum sequence length
MODEL_CHECKPOINT_DIR: "model_checkpoints/model.ckpt"  # Checkpoint directory
```

## Training

### Quick Start

```bash
python training.py
```

This will:
1. Load the custom tokenizer from `custom_enc.pkl`
2. Load and preprocess data from TinyStories
3. Initialize the MiniGPT model
4. Train with warmup cosine decay learning rate schedule
5. Save checkpoints using Orbax

### Training Features

- **Learning Rate Schedule**: Warmup (10% of total steps) followed by cosine decay
- **Loss Function**: Softmax cross-entropy with integer labels
- **Metrics**: Running average loss tracking via NNX metrics
- **Checkpointing**: Model saved periodically using Orbax for recovery

## Inference

### Text Generation

Run the inference script to generate text:

```bash
python inference.py
```

The inference pipeline supports:
- Custom text prompts
- Temperature-based sampling (controls randomness)
- Maximum token generation control
- Special token handling (e.g., `<|endoflabel|>`)

## Project Structure

```
├── config.yaml                    # Model and training configuration
├── data_loader.py                 # Data loading and preprocessing
├── model.py                       # MiniGPT architecture definition
├── training.py                    # Training script
├── inference.py                   # Text generation script
├── experiment.ipynb               # Jupyter notebook for experimentation
├── custom_enc.pkl                 # Custom tokenizer encoding
├── requirements.txt               # Python dependencies
└── small_checkpoint.orbax/        # Model checkpoints directory
```

## Dependencies

Key dependencies include:
- **JAX/Flax**: Neural network framework
- **Orbax**: Checkpointing and restoration
- **tiktoken**: Tokenization
- **Grain**: Efficient data loading
- **optax**: Optimization algorithms
- **PyYAML**: Configuration management

For the full list, see `requirements.txt`.

## Future Developments

- Training script
- Model checkpointing
- [ ] Generalized code with JAX wrapper to make it usable by others with different datasets
- [ ] Experiment tracking integration (MLflow)
- [ ] Model conversion to ONNX format
- [ ] Multi-device training

## License

See [LICENSE](LICENSE) for details.
