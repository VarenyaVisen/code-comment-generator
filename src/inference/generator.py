"""
Inference pipeline for generating code comments using the trained Transformer.
"""

import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.transformer import Transformer
from src.data_processing.tokenizer import SimpleTokenizer


class CodeCommentGenerator:
    """Generate comments for code using trained Transformer model."""
    
    def __init__(self, 
                 model_path: str = "checkpoints/best_model.pt",
                 tokenizer_path: str = "data/processed/tokenizer.pkl"):
        """
        Initialize the generator with trained model and tokenizer.
        
        Args:
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to saved tokenizer
        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Generator initializing on device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load(tokenizer_path)
        print(f"🔤 Tokenizer loaded: {self.tokenizer.vocab_size} tokens")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        print(f"🤖 Model loaded successfully!")
        print(f"📊 Model parameters: {self.model.count_parameters():,}")
        
        # Special token IDs
        self.start_token_id = self.tokenizer.token_to_id[self.tokenizer.start_token]
        self.end_token_id = self.tokenizer.token_to_id[self.tokenizer.end_token]
        self.pad_token_id = self.tokenizer.token_to_id[self.tokenizer.pad_token]
        
    def _load_model(self, model_path: str) -> Transformer:
        """Load the trained model from checkpoint."""
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model with same architecture
        model = Transformer(
            src_vocab_size=self.tokenizer.vocab_size,
            tgt_vocab_size=self.tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            d_ff=config['model']['d_ff'],
            max_seq_len=config['model']['max_seq_length'],
            dropout=config['model']['dropout']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
        print(f"📈 Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
    
    def generate_comment(self, 
                        code: str, 
                        max_length: int = 50,
                        temperature: float = 0.8,
                        top_k: int = 50,
                        top_p: float = 0.95) -> str:
        """
        Generate a comment for the given code.
        
        Args:
            code: Input code string
            max_length: Maximum length of generated comment
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated comment string
        """
        
        print(f"🔍 Generating comment for code:")
        print(f"   {code[:100]}{'...' if len(code) > 100 else ''}")
        
        with torch.no_grad():
            # Encode input code
            src_tokens = self.tokenizer.encode(code, is_code=True, max_length=50)
            src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
            
            print(f"📝 Encoded code: {len(src_tokens)} tokens")
            
            # Encode the source
            src_mask = (src_tensor != self.pad_token_id).unsqueeze(1).unsqueeze(2)
            encoder_output = self.model.encoder(src_tensor, src_mask)
            
            # Generate comment token by token
            generated_tokens = [self.start_token_id]
            
            for step in range(max_length):
                # Prepare current sequence
                tgt_tensor = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
                
                # Create causal mask
                tgt_len = tgt_tensor.size(1)
                tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Forward pass through decoder
                decoder_output = self.model.decoder(tgt_tensor, encoder_output, tgt_mask, src_mask)
                
                # Get next token probabilities
                next_token_logits = decoder_output[0, -1, :]  # Last token predictions
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k and top-p sampling
                next_token_id = self._sample_next_token(next_token_logits, top_k, top_p)
                
                # Add to sequence
                generated_tokens.append(next_token_id.item())
                
                # Stop if end token is generated
                if next_token_id.item() == self.end_token_id:
                    break
            
            # Decode generated tokens to text
            generated_text = self.tokenizer.decode(generated_tokens)
            
            print(f"✅ Generated comment: {generated_text}")
            return generated_text
    
    def _sample_next_token(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """
        Sample next token using top-k and top-p sampling.
        
        Args:
            logits: Token logits
            top_k: Top-k parameter
            top_p: Top-p parameter
            
        Returns:
            Sampled token ID
        """
        
        # Top-k sampling
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(0, top_k_indices, top_k_logits)
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def generate_multiple_comments(self, 
                                  code: str, 
                                  num_samples: int = 3,
                                  **kwargs) -> List[str]:
        """
        Generate multiple comment variations for the same code.
        
        Args:
            code: Input code string
            num_samples: Number of comments to generate
            **kwargs: Additional arguments for generate_comment
            
        Returns:
            List of generated comments
        """
        
        comments = []
        
        print(f"🎯 Generating {num_samples} comment variations...")
        
        for i in range(num_samples):
            print(f"\n🔄 Generating comment {i+1}/{num_samples}")
            comment = self.generate_comment(code, **kwargs)
            comments.append(comment)
        
        return comments
    
    def evaluate_on_test_examples(self):
        """Test the generator on some example functions."""
        
        test_examples = [
            """def calculate_area(radius):
    pi = 3.14159
    area = pi * radius * radius
    return area""",
            
            """def find_maximum(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers[1:]:
        if num > max_num:
            max_num = num
    return max_num""",
            
            """def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True"""
        ]
        
        print("🧪 Testing generator on example functions...\n")
        
        for i, code in enumerate(test_examples, 1):
            print(f"{'='*60}")
            print(f"📝 Test Example {i}:")
            print(f"{'='*60}")
            print("Code:")
            print(code)
            print("\n🤖 Generated Comment:")
            
            comment = self.generate_comment(
                code, 
                max_length=30,
                temperature=0.7,
                top_k=40
            )
            
            print(f"Result: {comment}")
            print()


def main():
    """Main inference function."""
    
    print("🚀 Code Comment Generator - Inference Pipeline\n")
    
    try:
        # Initialize generator
        generator = CodeCommentGenerator()
        
        # Test on examples
        generator.evaluate_on_test_examples()
        
        # Interactive mode
        print("🎯 Interactive Mode - Enter your code:")
        print("(Type 'quit' to exit)")
        
        while True:
            print("\n" + "="*50)
            code = input("Enter Python code: ").strip()
            
            if code.lower() == 'quit':
                break
            
            if code:
                comment = generator.generate_comment(code, temperature=0.7)
                print(f"\n🎉 Generated Comment: {comment}")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've trained the model first!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()