import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Register causal mask as a buffer so it's moved to the correct device
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)   # Project q, k, v together in a batch
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape dimensions for Multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # Use pre-registered mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

            
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.gelu = nn.GELU(approximate='tanh')
    
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()

        self.attention = CausalSelfAttention(seq_len, embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.feedforward = FeedForward(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x
    
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(seq_len, embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)  # Final LayerNorm
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0))  # Causal mask
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed_tokens(x) + self.pos_embeddings(pos)

        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # Logits for each token
