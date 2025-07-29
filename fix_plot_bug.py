"""
Fix the plotting bug in trainer.py
"""

def fix_plotting():
    with open("src/training/trainer.py", 'r') as f:
        content = f.read()
    
    # Find the plot_training_curves method and replace it
    old_plot_method = '''def plot_training_curves(self):
        """Plot training and validation loss curves."""
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        plt.plot(epochs, self.val_losses, label='Validation Loss', color='red')'''
    
    new_plot_method = '''def plot_training_curves(self):
        """Plot training and validation loss curves."""
        
        plt.figure(figsize=(10, 6))
        
        # Plot training loss for all epochs
        train_epochs = range(1, len(self.train_losses) + 1)
        plt.plot(train_epochs, self.train_losses, label='Training Loss', color='blue')
        
        # Plot validation loss only for epochs where it was calculated
        if self.val_losses:
            eval_every = self.config['training']['eval_every']
            val_epochs = range(eval_every, len(self.val_losses) * eval_every + 1, eval_every)
            plt.plot(val_epochs, self.val_losses, label='Validation Loss', color='red')'''
    
    content = content.replace(old_plot_method, new_plot_method)
    
    with open("src/training/trainer.py", 'w') as f:
        f.write(content)
    
    print("✅ Fixed plotting bug")

if __name__ == "__main__":
    fix_plotting()