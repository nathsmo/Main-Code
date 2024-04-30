import torch
import os

class ModelManager:
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def initialize(self):
        self.load_model()

    def load_model(self):
        print("Loading model...")

        latest_ckpt = self.find_latest_checkpoint(self.args['model_dir'])
        if latest_ckpt is not None:
            print(f"Loading model from {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model loaded successfully.")
        else:
            print("No checkpoint found. Initializing model with default weights.")

    def find_latest_checkpoint(self, directory):
        print(f"Looking for latest checkpoint in {directory}")
        checkpoint_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            return latest_checkpoint
        return None
    
    def save_model(model, optimizer, epoch, save_dir, file_name=None):
        """
        Saves the model and optimizer state at a given interval.
        
        Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        epoch (int): The current epoch number.
        save_dir (str): The directory where the model will be saved.
        file_name (str, optional): Optional custom file name for the checkpoint file. If None, uses a default naming convention.
        """
        if file_name is None:
            file_name = f'model_{epoch}.pth'  # Default naming convention
        save_path = os.path.join(save_dir, file_name)
        
        # Create directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Saving model and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        
        print(f"Model saved successfully at {save_path}")

