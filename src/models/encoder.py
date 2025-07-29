"""
Transformer Encoder implementation.
The encoder reads and understands the input code.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.attention import MultiHeadAttention, PositionalEncoding


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Hidden dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        print(f"🔧 FeedForward: {d_model} → {d_ff} → {d_model}")
    
    def forward(self, x):
        """
        Forward pass: x → ReLU(xW1) → xW2
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Structure:
    Input → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → Output
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"🏗️ EncoderLayer: d_model={d_model}, heads={n_heads}, d_ff={d_ff}")
    
    def forward(self, x, mask=None):
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        
        # 1. Multi-head self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-forward with residual connection and layer norm  
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder - stack of encoder layers.
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        """
        Initialize complete encoder.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"🏛️ TransformerEncoder: {n_layers} layers, vocab_size={vocab_size}")
    
    def forward(self, src, src_mask=None):
        """
        Forward pass through entire encoder.
        
        Args:
            src: Source tokens [batch, src_seq_len]
            src_mask: Source mask [batch, src_seq_len]
            
        Returns:
            Encoded representations [batch, src_seq_len, d_model]
        """
        
        # 1. Token embeddings + positional encoding
        x = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 2. Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x


def test_encoder():
    """Test the encoder implementation."""
    
    print("🧪 Testing Transformer Encoder...\n")
    
    # Test parameters
    vocab_size = 1000
    d_model = 128
    n_heads = 8
    n_layers = 4
    d_ff = 256
    max_seq_len = 64
    batch_size = 2
    seq_len = 10
    
    # Create test input (token IDs)
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {src.shape}")
    print(f"Sample input: {src[0][:5].tolist()}...")
    
    # Create encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    # Forward pass
    with torch.no_grad():
        encoded = encoder(src)
    
    print(f"\nEncoder output shape: {encoded.shape}")
    print(f"Expected shape: [batch={batch_size}, seq_len={seq_len}, d_model={d_model}]")
    
    print("✅ Encoder test passed!")
    return encoder


if __name__ == "__main__":
    test_encoder()