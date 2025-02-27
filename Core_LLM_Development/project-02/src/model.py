import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, type_vocab_size, seq_len, embed_dim, dropout_prob=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.segment_embedding = nn.Embedding(type_vocab_size, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, segment_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        embeddings = (self.token_embedding(input_ids) + 
                      self.position_embedding(position_ids) +
                      self.segment_embedding(segment_ids))
        
        return self.dropout(self.layer_norm(embeddings))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_prob=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out_proj(attn_output))

            
class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout_prob):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

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
    
class MLMHead(nn.Module):
    def __init__(self, embed_dim, vocab_size, token_embedding_weight):
        super().__init__()
        self.transform = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.decoder.weight = token_embedding_weight  # Tie weights with token embedding
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        x = self.transform(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias  # (batch_size, seq_length, vocab_size)
        return x
    
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, type_vocab_size, seq_len, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.embedding = Embedding(vocab_size, type_vocab_size, seq_len, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(seq_len, embed_dim, num_heads) for _ in range(num_layers)])
        self.lm_head = MLMHead( embed_dim, vocab_size, self.embedding.token_embedding.weight)

    def forward(self, token_ids, segment_ids, attention_mask):

        x = self.embedding(token_ids, segment_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.lm_head(x)  # Logits for each token
