"""
Test the training setup without running full training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.training.trainer import TransformerTrainer

def test_setup():
    """Test that everything is set up correctly."""
    
    print("🧪 Testing training setup...\n")
    
    try:
        # Create trainer
        trainer = TransformerTrainer()
        
        # Test data preparation
        trainer.prepare_data()
        print(f"✅ Data preparation successful")
        print(f"   Train samples: {len(trainer.train_dataset)}")
        print(f"   Val samples: {len(trainer.val_dataset)}")
        
        # Test model preparation
        trainer.prepare_model()
        print(f"✅ Model preparation successful")
        print(f"   Parameters: {trainer.model.count_parameters():,}")
        
        # Test one forward pass
        sample_batch = next(iter(trainer.train_loader))
        input_ids = sample_batch['input_ids'].to(trainer.device)
        target_ids = sample_batch['target_ids'].to(trainer.device)
        
        # Prepare decoder input
        decoder_input = target_ids[:, :-1]
        
        # Forward pass
        src_mask, tgt_mask = trainer.model.generate_masks(input_ids, decoder_input)
        outputs = trainer.model(input_ids, decoder_input, src_mask, tgt_mask)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {outputs.shape}")
        
        # Test loss calculation
        decoder_target = target_ids[:, 1:]
        loss = trainer.criterion(
            outputs.reshape(-1, outputs.size(-1)),
            decoder_target.reshape(-1)
        )
        
        print(f"✅ Loss calculation successful")
        print(f"   Loss value: {loss.item():.4f}")
        
        print(f"\n🎉 Training setup test passed!")
        print(f"🚀 Ready to start training!")
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_setup()