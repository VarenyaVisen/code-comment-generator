"""
Transformer Decoder implementation.
The decoder generates the output comment word by word.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.attention import MultiHeadAttention
from src.models.encoder import FeedForward


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Structure:
    Input → Self-Attention → Add & Norm → Cross-Attention → Add & Norm → Feed Forward → Add & Norm → Output
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Self-attention (looks at previously generated words)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention (looks at encoder output - the input code)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"🏗️ DecoderLayer: d_model={d_model}, heads={n_heads}, d_ff={d_ff}")
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Forward pass through decoder layer.
        
        Args:
            x: Target input [batch, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch, src_seq_len, d_model]
            tgt_mask: Target mask (prevents looking at future words)
            src_mask: Source mask (for padding)
            
        Returns:
            Output tensor [batch, tgt_seq_len, d_model]
        """
        
        # 1. Self-attention on target sequence (with causal mask)
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Cross-attention between target and source
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """
    Complete Transformer Decoder - stack of decoder layers.
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        """
        Initialize complete decoder.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        from .attention import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"🏛️ TransformerDecoder: {n_layers} layers, vocab_size={vocab_size}")
    
    def forward(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        Forward pass through entire decoder.
        
        Args:
            tgt: Target tokens [batch, tgt_seq_len]
            encoder_output: Encoder output [batch, src_seq_len, d_model]
            tgt_mask: Target mask (causal mask)
            src_mask: Source mask (padding mask)
            
        Returns:
            Output logits [batch, tgt_seq_len, vocab_size]
        """
        
        # 1. Token embeddings + positional encoding
        x = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 2. Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        # 3. Project to vocabulary size
        output = self.output_projection(x)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask to prevent looking at future tokens.
        
        Args:
            sz: Sequence length
            
        Returns:
            Mask tensor [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask


def test_decoder():
    """Test the decoder implementation."""
    
    print("🧪 Testing Transformer Decoder...\n")
    
    # Test parameters
    vocab_size = 1000
    d_model = 128
    n_heads = 8
    n_layers = 4
    d_ff = 256
    max_seq_len = 64
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # Create test inputs
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)
    
    print(f"Target input shape: {tgt.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Create decoder
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    # Create causal mask
    tgt_mask = decoder.generate_square_subsequent_mask(tgt_seq_len)
    print(f"Causal mask shape: {tgt_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = decoder(tgt, encoder_output, tgt_mask)
    
    print(f"\nDecoder output shape: {output.shape}")
    print(f"Expected shape: [batch={batch_size}, tgt_seq_len={tgt_seq_len}, vocab_size={vocab_size}]")
    
    print("✅ Decoder test passed!")
    return decoder


if __name__ == "__main__":
    test_decoder()