"""
Simple tokenizer for code-comment pairs.
Converts text to numbers that the Transformer can understand.
"""

import json
import yaml
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import Counter
import pickle


class SimpleTokenizer:
    """A simple word-level tokenizer for code and comments."""
    
    def __init__(self, config_path: str = "configs/tokenizer_config.yaml"):
        """Initialize tokenizer with configuration."""
        self.config = self._load_config(config_path)
        
        # Special tokens
        self.pad_token = self.config["tokenizer"]["pad_token"]
        self.unk_token = self.config["tokenizer"]["unk_token"] 
        self.start_token = self.config["tokenizer"]["start_token"]
        self.end_token = self.config["tokenizer"]["end_token"]
        
        # Vocabularies (will be built from data)
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
        print("🔤 Tokenizer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load tokenizer configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _tokenize_text(self, text: str, is_code: bool = False) -> List[str]:
        """Convert text into tokens (words/subwords)."""
        
        # Clean the text
        text = text.strip()
        
        if is_code:
            # Special handling for code
            # Replace common patterns with readable tokens
            text = re.sub(r'\n    ', ' <INDENT> ', text)  # Indentation
            text = re.sub(r'\n', ' <NEWLINE> ', text)     # Line breaks
            text = re.sub(r'  +', ' ', text)              # Multiple spaces
        
        # Simple word tokenization
        # Split on whitespace and punctuation, but keep important chars
        tokens = []
        
        # Split by spaces first
        words = text.split()
        
        for word in words:
            # Further split on punctuation while keeping it
            sub_tokens = re.findall(r'\w+|[^\w\s]', word)
            tokens.extend(sub_tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocabulary(self, dataset: List[Dict]) -> None:
        """Build vocabulary from the dataset."""
        
        print("🏗️ Building vocabulary from dataset...")
        
        # Collect all tokens
        all_tokens = []
        
        for example in dataset:
            # Tokenize code
            code_tokens = self._tokenize_text(example["function_code"], is_code=True)
            all_tokens.extend(code_tokens)
            
            # Tokenize comment  
            comment_tokens = self._tokenize_text(example["docstring"], is_code=False)
            all_tokens.extend(comment_tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        print(f"📊 Found {len(token_counts)} unique tokens")
        print(f"🔤 Most common tokens: {token_counts.most_common(10)}")
        
        # Build vocabulary: start with special tokens
        vocab = [
            self.pad_token,
            self.unk_token, 
            self.start_token,
            self.end_token
        ]
        
        # Add special tokens from config
        special_tokens = self.config["tokenizer"]["special_tokens"]
        vocab.extend(special_tokens)
        
        # Add most common tokens up to vocab_size
        max_vocab_size = self.config["tokenizer"]["vocab_size"]
        remaining_slots = max_vocab_size - len(vocab)
        
        most_common = token_counts.most_common(remaining_slots)
        vocab.extend([token for token, count in most_common])
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        print(f"✅ Vocabulary built: {self.vocab_size} tokens")
        print(f"📋 Special tokens: {vocab[:8]}")  # Show first 8 tokens
        
        # Show some example mappings
        print(f"🔍 Example mappings:")
        for i, token in enumerate(vocab[:15]):
            print(f"  '{token}' → {i}")
    
    def encode(self, text: str, is_code: bool = False, max_length: Optional[int] = None) -> List[int]:
        """Convert text to token IDs."""
        
        # Tokenize
        tokens = self._tokenize_text(text, is_code=is_code)
        
        # Add start and end tokens
        tokens = [self.start_token] + tokens + [self.end_token]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Unknown token
                token_ids.append(self.token_to_id[self.unk_token])
        
        # Apply length limit
        if max_length:
            if len(token_ids) > max_length:
                # Truncate but keep end token
                token_ids = token_ids[:max_length-1] + [self.token_to_id[self.end_token]]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip padding tokens in output
                if token != self.pad_token:
                    tokens.append(token)
        
        # Join tokens back to text
        text = " ".join(tokens)
        
        # Clean up formatting
        text = text.replace(f" {self.start_token} ", "")
        text = text.replace(f" {self.end_token} ", "")
        text = text.replace(" <NEWLINE> ", "\n")
        text = text.replace(" <INDENT> ", "    ")
        
        return text.strip()
    
    def pad_sequence(self, token_ids: List[int], max_length: int) -> List[int]:
        """Pad sequence to fixed length."""
        
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            # Pad with pad_token_id
            pad_id = self.token_to_id[self.pad_token]
            padding = [pad_id] * (max_length - len(token_ids))
            return token_ids + padding
    
    def save(self, save_path: str) -> None:
        """Save tokenizer to file."""
        
        tokenizer_data = {
            "config": self.config,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "vocab_size": self.vocab_size
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"💾 Tokenizer saved to {save_path}")
    
    def load(self, load_path: str) -> None:
        """Load tokenizer from file."""
        
        with open(load_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.config = tokenizer_data["config"]
        self.token_to_id = tokenizer_data["token_to_id"]
        self.id_to_token = tokenizer_data["id_to_token"]
        self.vocab_size = tokenizer_data["vocab_size"]
        
        print(f"📂 Tokenizer loaded from {load_path}")


def test_tokenizer():
    """Test the tokenizer with our dataset."""
    
    print("🧪 Testing tokenizer with our curated dataset...\n")
    
    # Load our data
    with open("data/raw/code_comments_dataset.json", 'r') as f:
        dataset = json.load(f)
    
    print(f"📂 Loaded {len(dataset)} examples")
    
    # Create and train tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocabulary(dataset)
    
    # Test with first example
    sample = dataset[0]
    code = sample["function_code"]
    comment = sample["docstring"]
    
    print(f"\n" + "="*60)
    print(f"📝 Testing with sample:")
    print(f"Original code:\n{code}")
    print(f"\nOriginal comment:\n{comment}")
    
    # Encode
    print(f"\n🔢 Encoding...")
    code_ids = tokenizer.encode(code, is_code=True)
    comment_ids = tokenizer.encode(comment, is_code=False)
    
    print(f"Code tokens: {len(code_ids)} tokens")
    print(f"Code IDs: {code_ids}")
    print(f"Comment tokens: {len(comment_ids)} tokens") 
    print(f"Comment IDs: {comment_ids}")
    
    # Decode back
    print(f"\n🔄 Decoding back to text...")
    decoded_code = tokenizer.decode(code_ids)
    decoded_comment = tokenizer.decode(comment_ids)
    
    print(f"Decoded code:\n{decoded_code}")
    print(f"\nDecoded comment:\n{decoded_comment}")
    
    # Test padding
    print(f"\n📏 Testing padding...")
    max_len = 25
    padded_code = tokenizer.pad_sequence(code_ids, max_len)
    print(f"Original length: {len(code_ids)}")
    print(f"Padded length: {len(padded_code)}")
    print(f"Padded sequence: {padded_code}")
    
    print(f"\n✅ Tokenizer test complete!")
    return tokenizer


if __name__ == "__main__":
    tokenizer = test_tokenizer()