"""
Complete Transformer model for Code Comment Generation.
Combines encoder and decoder into a full sequence-to-sequence model.
"""

import torch
import torch.nn as nn
import yaml
from typing import Optional, Tuple
import sys
import os

# Fix import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our components
from src.models.encoder import TransformerEncoder
from src.models.decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Complete Transformer model for generating comments from code.
    
    Architecture:
    Code → Encoder → Memory → Decoder → Comment
    """
    
    def __init__(self, 
                 src_vocab_size: int,     # Size of code vocabulary
                 tgt_vocab_size: int,     # Size of comment vocabulary  
                 d_model: int = 128,      # Hidden dimension
                 n_heads: int = 8,        # Number of attention heads
                 n_layers: int = 4,       # Number of encoder/decoder layers
                 d_ff: int = 256,         # Feed-forward dimension
                 max_seq_len: int = 64,   # Maximum sequence length
                 dropout: float = 0.1):   # Dropout rate
        """
        Initialize the complete Transformer.
        
        Args:
            src_vocab_size: Code vocabulary size
            tgt_vocab_size: Comment vocabulary size (usually same as src)
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of layers in encoder/decoder
            d_ff: Feed-forward network dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Encoder: Processes input code
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Decoder: Generates output comments
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        print(f"🤖 Complete Transformer initialized!")
        print(f"   📊 Parameters: src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}")
        print(f"   🏗️ Architecture: {n_layers} layers, {n_heads} heads, d_model={d_model}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the complete Transformer.
        
        Args:
            src: Source tokens (code) [batch, src_seq_len]
            tgt: Target tokens (comment) [batch, tgt_seq_len]
            src_mask: Source padding mask [batch, src_seq_len]
            tgt_mask: Target causal mask [tgt_seq_len, tgt_seq_len]
            
        Returns:
            Output logits [batch, tgt_seq_len, tgt_vocab_size]
        """
        
        # 1. Encode the source (code)
        encoder_output = self.encoder(src, src_mask)
        
        # 2. Decode to generate target (comment)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        return decoder_output
    
    def generate_masks(self, src, tgt):
        """
        Generate attention masks for source and target sequences.
        
        Args:
            src: Source tokens [batch, src_seq_len]
            tgt: Target tokens [batch, tgt_seq_len]
            
        Returns:
            src_mask: Source padding mask
            tgt_mask: Target causal mask
        """
        
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        # Source mask: Hide padding tokens (assuming 0 is padding token)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_seq_len]
        
        # Target causal mask: Prevent looking at future tokens
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len) * float('-inf'), diagonal=1)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_seq_len, tgt_seq_len]
        
        # Target padding mask
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_seq_len]
        
        # Combine causal and padding masks for target
        tgt_mask = tgt_mask + (tgt_padding_mask == 0).float() * float('-inf')
        
        return src_mask, tgt_mask
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerConfig:
    """Configuration class for Transformer model."""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Model parameters
        self.d_model = config['model']['d_model']
        self.d_ff = config['model']['d_ff']
        self.n_heads = config['model']['n_heads']
        self.n_layers = config['model']['n_layers']
        self.vocab_size = config['model']['vocab_size']
        self.max_seq_length = config['model']['max_seq_length']
        self.dropout = config['model']['dropout']
        self.attention_dropout = config['model']['attention_dropout']
        
        # Training parameters
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.num_epochs = config['training']['num_epochs']
        self.warmup_steps = config['training']['warmup_steps']
        
        print(f"📋 Configuration loaded from {config_path}")


def create_model_from_config(config_path: str = "configs/model_config.yaml") -> Transformer:
    """
    Create Transformer model from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Transformer model
    """
    
    config = TransformerConfig(config_path)
    
    # Update vocab size from processed data if available
    try:
        import json
        with open("data/processed/dataset_info.json", 'r') as f:
            dataset_info = json.load(f)
        vocab_size = dataset_info['vocab_size']
        print(f"📊 Updated vocab_size from dataset: {vocab_size}")
    except FileNotFoundError:
        vocab_size = config.vocab_size
        print(f"⚠️ Using config vocab_size: {vocab_size}")
    
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,  # Same vocabulary for code and comments
        d_model=config.d_model,
        n_heads=config.n_heads, 
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_length,
        dropout=config.dropout
    )
    
    return model


def test_transformer():
    """Test the complete Transformer model."""
    
    print("🧪 Testing Complete Transformer Model...\n")
    
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
    
    # Create model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    print(f"📊 Model has {model.count_parameters():,} trainable parameters")
    
    # Create test data
    src = torch.randint(1, vocab_size, (batch_size, src_seq_len))  # Avoid 0 (padding)
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\n📥 Input shapes:")
    print(f"   Source (code): {src.shape}")
    print(f"   Target (comment): {tgt.shape}")
    
    # Generate masks
    src_mask, tgt_mask = model.generate_masks(src, tgt)
    print(f"   Source mask: {src_mask.shape}")
    print(f"   Target mask: {tgt_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\n📤 Output shape: {output.shape}")
    print(f"   Expected: [batch={batch_size}, tgt_seq_len={tgt_seq_len}, vocab_size={vocab_size}]")
    
    # Test output properties
    print(f"\n🔍 Output analysis:")
    print(f"   Min value: {output.min().item():.3f}")
    print(f"   Max value: {output.max().item():.3f}")
    print(f"   Mean: {output.mean().item():.3f}")
    
    # Test that we can get probabilities
    probs = torch.softmax(output, dim=-1)
    print(f"   Probability sum (should be ~1.0): {probs.sum(dim=-1).mean().item():.3f}")
    
    print("\n✅ Complete Transformer test passed!")
    return model


def test_with_real_data():
    """Test the model with our actual processed data."""
    
    print("🔬 Testing with Real Processed Data...\n")
    
    try:
        # Load dataset info
        import json
        with open("data/processed/dataset_info.json", 'r') as f:
            dataset_info = json.load(f)
        
        vocab_size = dataset_info['vocab_size']
        max_code_len = dataset_info['max_code_length']
        max_comment_len = dataset_info['max_comment_length']
        
        print(f"📊 Real dataset parameters:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Max code length: {max_code_len}")
        print(f"   Max comment length: {max_comment_len}")
        
        # Create model with real parameters
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=256,
            max_seq_len=max(max_code_len, max_comment_len)
        )
        
        print(f"🤖 Real model has {model.count_parameters():,} parameters")
        
        # Test with realistic sequence lengths
        batch_size = 2
        src = torch.randint(1, vocab_size, (batch_size, max_code_len))
        tgt = torch.randint(1, vocab_size, (batch_size, max_comment_len))
        
        # Forward pass
        with torch.no_grad():
            output = model(src, tgt)
        
        print(f"✅ Real data test passed! Output shape: {output.shape}")
        
        return model
        
    except FileNotFoundError:
        print("❌ Processed data not found. Run dataset_processor.py first.")
        return None


if __name__ == "__main__":
    print("🚀 Testing Complete Transformer Implementation\n")
    print("="*60)
    
    # Test 1: Basic functionality
    model1 = test_transformer()
    
    print("\n" + "="*60)
    
    # Test 2: With real data parameters
    model2 = test_with_real_data()
    
    print("\n🎉 All tests completed!")