"""
Multi-Head Attention implementation - the core of the Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from "Attention Is All You Need".
    
    This is the core innovation of the Transformer architecture.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Hidden dimension size
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        print(f"🔍 MultiHeadAttention: {n_heads} heads, {self.d_k} dims per head")
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection  
        self.w_v = nn.Linear(d_model, d_model)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        
        Args:
            Q: Query tensor [batch, heads, seq_len, d_k]
            K: Key tensor [batch, heads, seq_len, d_k]  
            V: Value tensor [batch, heads, seq_len, d_k]
            mask: Optional mask tensor
            
        Returns:
            attention_output: [batch, heads, seq_len, d_k]
            attention_weights: [batch, heads, seq_len, seq_len]
        """
        
        # Calculate attention scores
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for padding or future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of Multi-Head Attention.
    
        Args:
            query: Query tensor [batch, query_seq_len, d_model]
            key: Key tensor [batch, key_seq_len, d_model]
            value: Value tensor [batch, value_seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            output: [batch, query_seq_len, d_model]
            attention_weights: [batch, n_heads, query_seq_len, key_seq_len]
        """
    
        batch_size = query.size(0)
        query_seq_len = query.size(1)
        key_seq_len = key.size(1)
        value_seq_len = value.size(1)
    
        # 1. Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, query_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value_seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
        # 2. Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
    
        # 3. Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_seq_len, self.d_model
        )
    
        # 4. Final linear projection
        output = self.w_o(attention_output)
    
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """
    Positional encoding to inject sequence order information.
    
    Uses sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 512):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Hidden dimension size
            max_len: Maximum sequence length
        """
        super().__init__()
        
        print(f"📍 PositionalEncoding: max_len={max_len}, d_model={d_model}")
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even positions, cosine to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]


def test_attention():
    """Test the attention mechanism."""
    
    print("🧪 Testing Multi-Head Attention...\n")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 128
    n_heads = 8
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Test Multi-Head Attention
    attention = MultiHeadAttention(d_model, n_heads)
    output, weights = attention(x, x, x)  # Self-attention
    
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Test Positional Encoding
    pos_encoding = PositionalEncoding(d_model)
    x_with_pos = pos_encoding(x)
    
    print(f"With positional encoding: {x_with_pos.shape}")
    
    print("✅ Attention mechanism test passed!")


if __name__ == "__main__":
    test_attention()