import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        embeddings = (self.token_embeddings(input_ids) + 
                      self.position_embeddings(position_ids) +
                      self.segment_embeddings(token_type_ids))
        
        return self.dropout(self.layer_norm(embeddings))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_prob=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)  # Single projection for Q, K, V
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # Final projection layer
        self.dropout = nn.Dropout(dropout_prob)  # Dropout for regularization

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape  # B: Batch size, T: Sequence length, C: Embedding dimension

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        # Apply attention mask (for padding tokens)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)  # Apply dropout
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

        self.attention = MultiHeadSelfAttention(seq_len, embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.feedforward = FeedForward(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x
    
class BERTModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(seq_len, embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)  # Final LayerNorm
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0))  # Causal mask

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed_tokens(x) + self.pos_embeddings(pos)

        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # Logits for each token
