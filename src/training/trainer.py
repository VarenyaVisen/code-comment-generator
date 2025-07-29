"""
Training pipeline for the Transformer model.
Handles training loop, evaluation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.transformer import create_model_from_config
from src.data_processing.dataset_processor import DatasetProcessor
from src.data_processing.tokenizer import SimpleTokenizer


class TransformerTrainer:
    """Handles training and evaluation of the Transformer model."""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize trainer with configuration."""
        
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        
        print(f"🚀 Trainer initialized on device: {self.device}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device (GPU/CPU)."""
        if self.config['device'] == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                print("💻 Using CPU")
        else:
            device = torch.device(self.config['device'])
        
        return device
    
    def _create_directories(self):
        """Create necessary directories for saving models and logs."""
        Path(self.config['paths']['model_save_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['paths']['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self):
        """Load and prepare training data."""
        
        print("📊 Preparing training data...")
        
        # Load tokenizer
        tokenizer_path = Path(self.config['paths']['processed_data']) / "tokenizer.pkl"
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load(str(tokenizer_path))
        
        print(f"🔤 Tokenizer loaded: {self.tokenizer.vocab_size} tokens")
        
        # Load datasets
        processor = DatasetProcessor()
        self.train_dataset, self.val_dataset, _ = processor.process_dataset()
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        print(f"📈 Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
        print(f"🔄 Batches: {len(self.train_loader)} train, {len(self.val_loader)} val")
    
    def _collate_fn(self, batch):
        """Collate function for data loader."""
        
        # Extract sequences
        input_ids = torch.stack([item['input_ids'] for item in batch])
        target_ids = torch.stack([item['target_ids'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }
    
    def prepare_model(self):
        """Initialize model, optimizer, and loss function."""
        
        print("🤖 Preparing model...")
        
        # Create model
        self.model = create_model_from_config()
        self.model.to(self.device)
        
        print(f"📊 Model parameters: {self.model.count_parameters():,}")
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignore padding tokens
            label_smoothing=self.config['training']['label_smoothing']
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['adam_beta1'], self.config['training']['adam_beta2']),
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            # verbose=True
        )
        
        print("✅ Model, optimizer, and loss function ready")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Prepare decoder input (shifted by one position)
            decoder_input = target_ids[:, :-1]  # Remove last token
            decoder_target = target_ids[:, 1:]  # Remove first token
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Generate masks
            src_mask, tgt_mask = self.model.generate_masks(input_ids, decoder_input)
            
            # Model forward
            outputs = self.model(input_ids, decoder_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                decoder_target.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip_norm']
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{avg_loss:.4f}'
            })
        
        return total_loss / num_batches
    
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Prepare decoder input
                decoder_input = target_ids[:, :-1]
                decoder_target = target_ids[:, 1:]
                
                # Forward pass
                src_mask, tgt_mask = self.model.generate_masks(input_ids, decoder_input)
                outputs = self.model(input_ids, decoder_input, src_mask, tgt_mask)
                
                # Calculate loss
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    decoder_target.reshape(-1)
                )
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config['paths']['model_save_dir']) / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config['paths']['model_save_dir']) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        
        plt.figure(figsize=(10, 6))
        
        # Plot training loss for all epochs
        train_epochs = range(1, len(self.train_losses) + 1)
        plt.plot(train_epochs, self.train_losses, label='Training Loss', color='blue')
        
        # Plot validation loss only for epochs where it was calculated
        if self.val_losses:
            eval_every = self.config['training']['eval_every']
            val_epochs = range(eval_every, len(self.val_losses) * eval_every + 1, eval_every)
            plt.plot(val_epochs, self.val_losses, label='Validation Loss', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = Path(self.config['paths']['logs_dir']) / "training_curves.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"📊 Training curves saved: {plot_path}")
    
    def train(self):
        """Main training loop."""
        
        print("🚀 Starting training...\n")
        
        num_epochs = self.config['training']['num_epochs']
        eval_every = self.config['training']['eval_every']
        save_every = self.config['training']['save_every']
        patience = self.config['training']['patience']
        
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Evaluation
            if epoch % eval_every == 0:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                
                print(f"  Val Loss: {val_loss:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.config['training']['min_delta']:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += eval_every
                
                # Early stopping
                if self.patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(is_best=False)
            
            print()
        
        # Final save and plot
        self.save_checkpoint(is_best=False)
        self.plot_training_curves()
        
        print("🎉 Training completed!")
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training function."""
    
    print("🚀 Code Comment Generator Training\n")
    
    # Create trainer
    trainer = TransformerTrainer()
    
    # Prepare data and model
    trainer.prepare_data()
    trainer.prepare_model()
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()