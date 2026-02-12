import torch
import torch.nn as nn
from torch.nn import functional as F

# Head Python class
class Head(nn.Module):
    """One head of self-attention"""

    # Head class contructor
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.block_size = block_size
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = dropout
        self.dropout_head = nn.Dropout(self.dropout)

    # Controls the forward pass of the Head self attention block
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)         # (B, T, C)
        q = self.query(x)       # (B, T, C)
        # Compute the attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**-0.5      # (B, T, C) @ (B, C, T) -. (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout_head(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)       # (B, T, C)
        out = wei @ v       # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# MultiHeadAttention Python class
class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention in paralell"""

    # MultiHeadAttention class contructor
    def __init__(self, n_embd, n_heads, head_size, block_size, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = dropout
        self.heads = nn.ModuleList([Head(n_embd=self.n_embd, head_size=self.head_size, block_size=self.block_size, dropout=self.dropout) for _ in range(self.n_heads)])
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout_attn = nn.Dropout(self.dropout)

    # Controls the forward pass of the multihead attention block
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout_attn(self.proj(out))            # Projection back into the residual pathway
        return out

# FeedForward Python class object
class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    # FeedForward Python class constructor
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.dropout = dropout

        self.net= nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.ReLU(),
            nn.Linear(4 * self.n_embd, self.n_embd),      # Projection layer going back into the residual pathway
            nn.Dropout(self.dropout),
        )

    # Controls the forward pass of the Feed Forward neural network layers
    def forward(self, x):
        return self.net(x)

# Transformer Block Python class
class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    # Transformer Block class constructor
    def __init__(self, n_embd, n_head, block_size, dropout):
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout

        # n_emdb: embeding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = self.n_embd // self.n_head
        self.sa = MultiHeadAttention(self.n_embd, self.n_head, head_size, self.block_size, self.dropout)
        self.ffwd = FeedForward(self.n_embd, self.dropout)
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)

    # Controls the forward pass of the transformer block
    def forward(self, x):
        """Forward method of the Blobk object with residual connections"""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# BigramLanguageModel Python class object
class BigramLanguageModel(nn.Module):

    """Bigram Lamguage model"""

    # BigramLanguageModel class constructor
    def __init__(self, vocab_size, device, cfg):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.cfg = cfg

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, cfg.training.n_embd)       # replaced vocab_size with n_embd
        self.position_embedding_table = nn.Embedding(cfg.training.block_size, cfg.training.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg.training.n_embd, n_head=cfg.training.n_head, block_size=cfg.training.block_size, dropout=cfg.training.dropout) for _ in range(cfg.training.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.training.n_embd)   # Final LayerNorm
        self.lm_head = nn.Linear(cfg.training.n_embd, self.vocab_size)

    # Controls the forward pass of the Bigram Language Model
    def forward(self, idx, targets=None):
        # Decode B and t from idx
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)    # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))    # (T, C)
        x = tok_emb + pos_emb           # (B, T, C)
        x = self.blocks(x)      # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # If targets is equal to None
        if targets is None:

            # Sets the loss variable to None
            loss = None

        # When targets has values
        else:

            # Reshapes the logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # calculates the cross entropy loss with PyTorch
            loss = F.cross_entropy(logits, targets)

        # Returns the logits and the loss from the model
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indicies in hte current context
        for _ in range(max_new_tokens):

            # idx is (B, T) array of indices in the current context
            idx_cond = idx[:, -self.cfg.training.block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # Focus only on hte last time step
            logits = logits[:, -1, :]   # becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)      # (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)   # (B, T+1)

        return idx