import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx

class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)

def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs
        )
        # Feed-forward network: embed_dim -> ff_dim -> embed_dim
        self.ff_dense1 = nnx.Linear(embed_dim, ff_dim, rngs=rngs)
        self.ff_dense2 = nnx.Linear(ff_dim, embed_dim, rngs=rngs)
        
    def __call__(self, x, mask=None):
        # Attention with residual connection
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ff_out = self.ff_dense1(x)
        ff_out = jnn.relu(ff_out)  # ReLU activation
        ff_out = self.ff_dense2(ff_out)
        x = x + ff_out
        
        return x

class MiniGPT(nnx.Module):

    def __init__(self, maxlen, vocab_size, embed_dim, num_heads,
                 feed_forward_dim, num_transformer_blocks, *, rngs):
        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, 
                                                   embed_dim, rngs=rngs)
        self.transformer_blocks: list = [
    TransformerBlock(embed_dim, num_heads, feed_forward_dim, 
                     rngs=rngs)
    for _ in range(num_transformer_blocks)
]
        self.output_layer = nnx.Linear(embed_dim, vocab_size, 
                                       use_bias=False, rngs=rngs)
        
    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)
        return logits