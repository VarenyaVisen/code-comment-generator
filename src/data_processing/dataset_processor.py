"""
Process the raw dataset into training-ready format.
"""

import json
import yaml
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
import sys
import os

# Fix import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_processing.tokenizer import SimpleTokenizer


class CodeCommentDataset(Dataset):
    """PyTorch dataset for code-comment pairs."""
    
    def __init__(self, examples: List[Dict], tokenizer: SimpleTokenizer, max_code_len: int = 50, max_comment_len: int = 30):
        """Initialize dataset."""
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_comment_len = max_comment_len
        
        print(f"📊 Dataset created with {len(examples)} examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Encode sequences
        code_ids = self.tokenizer.encode(example["function_code"], is_code=True, max_length=self.max_code_len)
        comment_ids = self.tokenizer.encode(example["docstring"], is_code=False, max_length=self.max_comment_len)
        
        # Pad sequences
        code_ids = self.tokenizer.pad_sequence(code_ids, self.max_code_len)
        comment_ids = self.tokenizer.pad_sequence(comment_ids, self.max_comment_len)
        
        return {
            "input_ids": torch.tensor(code_ids, dtype=torch.long),      # Encoder input (code)
            "target_ids": torch.tensor(comment_ids, dtype=torch.long),  # Decoder target (comment)
            "code_text": example["function_code"],                      # For reference
            "comment_text": example["docstring"]                       # For reference
        }


class DatasetProcessor:
    """Process raw data into training-ready datasets."""
    
    def __init__(self, config_path: str = "configs/tokenizer_config.yaml"):
        """Initialize processor."""
        self.config = self._load_config(config_path)
        self.tokenizer = None
        
        # Create output directories
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def process_dataset(self, raw_data_path: str = "data/raw/code_comments_dataset.json"):
        """Process the complete dataset."""
        
        print("🔄 Processing dataset for training...\n")
        
        # Load raw data
        with open(raw_data_path, 'r') as f:
            raw_data = json.load(f)
        
        print(f"📂 Loaded {len(raw_data)} examples from {raw_data_path}")
        
        # Create and train tokenizer
        print(f"\n🔤 Building tokenizer...")
        self.tokenizer = SimpleTokenizer(config_path="configs/tokenizer_config.yaml")
        self.tokenizer.build_vocabulary(raw_data)
        
        # Save tokenizer
        tokenizer_path = self.processed_dir / "tokenizer.pkl"
        self.tokenizer.save(str(tokenizer_path))
        
        # Split data
        print(f"\n📊 Splitting data...")
        train_data, val_data = self._split_data(raw_data)
        
        # Create datasets
        max_code_len = self.config["processing"]["max_code_tokens"]
        max_comment_len = self.config["processing"]["max_comment_tokens"]
        
        train_dataset = CodeCommentDataset(train_data, self.tokenizer, max_code_len, max_comment_len)
        val_dataset = CodeCommentDataset(val_data, self.tokenizer, max_code_len, max_comment_len)
        
        # Save processed data
        self._save_processed_data(train_data, val_data, train_dataset, val_dataset)
        
        # Show statistics
        self._show_statistics(train_dataset, val_dataset)
        
        return train_dataset, val_dataset, self.tokenizer
    
    def _split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train/validation sets."""
        
        train_ratio = self.config["data_split"]["train_ratio"]
        
        # For small datasets, use simple split
        split_idx = max(1, int(len(data) * train_ratio))  # At least 1 for training
        train_data = data[:split_idx]
        val_data = data[split_idx:] if split_idx < len(data) else [data[-1]]  # At least 1 for validation
        
        print(f"📊 Split: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    
    def _save_processed_data(self, train_data: List[Dict], val_data: List[Dict], 
                           train_dataset: CodeCommentDataset, val_dataset: CodeCommentDataset):
        """Save processed data."""
        
        # Save raw splits
        with open(self.processed_dir / "train_data.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(self.processed_dir / "val_data.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        # Save dataset info
        dataset_info = {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "vocab_size": self.tokenizer.vocab_size,
            "max_code_length": self.config["processing"]["max_code_tokens"],
            "max_comment_length": self.config["processing"]["max_comment_tokens"],
            "special_tokens": {
                "pad": self.tokenizer.pad_token,
                "unk": self.tokenizer.unk_token,
                "start": self.tokenizer.start_token,
                "end": self.tokenizer.end_token
            }
        }
        
        with open(self.processed_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"💾 Processed data saved to {self.processed_dir}")
    
    def _show_statistics(self, train_dataset: CodeCommentDataset, val_dataset: CodeCommentDataset):
        """Show dataset statistics."""
        
        print(f"\n📈 Dataset Statistics:")
        print(f"├── Training examples: {len(train_dataset)}")
        print(f"├── Validation examples: {len(val_dataset)}")
        print(f"├── Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"├── Max code length: {self.config['processing']['max_code_tokens']} tokens")
        print(f"└── Max comment length: {self.config['processing']['max_comment_tokens']} tokens")
        
        # Show a sample processed example
        sample = train_dataset[0]
        print(f"\n📝 Sample processed example:")
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Target shape: {sample['target_ids'].shape}")
        print(f"Code tokens: {sample['input_ids'][:15].tolist()}...")
        print(f"Comment tokens: {sample['target_ids'][:15].tolist()}...")
        print(f"Original code: {sample['code_text'][:60]}...")
        print(f"Original comment: {sample['comment_text'][:60]}...")


def main():
    """Main processing function."""
    
    print("🚀 Starting data preprocessing...\n")
    
    # Process the dataset
    processor = DatasetProcessor()
    train_dataset, val_dataset, tokenizer = processor.process_dataset()
    
    print(f"\n✅ Data preprocessing complete!")
    print(f"🎯 Ready for Transformer training!")
    
    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    main()