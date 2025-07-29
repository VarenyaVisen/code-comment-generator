"""
Manually plot training results.
"""

import matplotlib.pyplot as plt
import torch
from pathlib import Path

def plot_results():
    """Plot training results from the saved checkpoint."""
    
    try:
        # Load the best model checkpoint
        checkpoint_path = "checkpoints/best_model.pt"
        if not Path(checkpoint_path).exists():
            # Try to find any checkpoint
            checkpoint_dir = Path("checkpoints")
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            else:
                print("❌ No checkpoints found")
                return
        
        print(f"📊 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
        print(f"📈 Training data:")
        print(f"   Epochs trained: {len(train_losses)}")
        print(f"   Validation points: {len(val_losses)}")
        print(f"   Final train loss: {train_losses[-1]:.4f}")
        print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        train_epochs = range(1, len(train_losses) + 1)
        plt.plot(train_epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
        
        # Plot validation loss (every 10 epochs)
        if val_losses:
            val_epochs = range(10, len(val_losses) * 10 + 1, 10)
            plt.plot(val_epochs[:len(val_losses)], val_losses, label='Validation Loss', color='red', linewidth=2, marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Code Comment Generator - Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.annotate(f'Final Train Loss: {train_losses[-1]:.3f}', 
                    xy=(len(train_losses), train_losses[-1]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.1))
        
        if val_losses:
            best_val_idx = val_losses.index(min(val_losses))
            best_val_epoch = (best_val_idx + 1) * 10
            plt.annotate(f'Best Val Loss: {min(val_losses):.3f}', 
                        xy=(best_val_epoch, min(val_losses)), 
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1))
        
        # Save plot
        plt.tight_layout()
        plot_path = "logs/training_results.png"
        Path("logs").mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Plot saved to: {plot_path}")
        
        # Print training summary
        print(f"\n🎯 Training Summary:")
        print(f"   Started with loss: ~5.24")
        print(f"   Ended with loss: {train_losses[-1]:.4f}")
        print(f"   Improvement: {((5.24 - train_losses[-1]) / 5.24 * 100):.1f}%")
        print(f"   Model saved to: checkpoints/best_model.pt")
        
    except Exception as e:
        print(f"❌ Error plotting results: {e}")

if __name__ == "__main__":
    plot_results()