import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import time
import sys

print_grad = True

class GradientPrinter(Callback):
    def on_after_backward(self, trainer, pl_module):
        if print_grad:
            print("-----------------------------------")
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    print(f'{name: <50} -- value: {param.data.norm():.12f} -- grad: {param.grad.data.norm():.12f}')
                else:
                    print(f'{name: <50} -- value: {param.data.norm():.12f}')
            print("-----------------------------------")

class Logger(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.log_dir = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.logger = TensorBoardLogger(self.log_dir)

    def training_step(self, batch, batch_idx):
        # Example training step
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def compute_loss(self, batch):
        # Placeholder for actual loss computation
        return torch.tensor(0.0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1),
                "monitor": "train_loss",
            },
        }

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def gradient_clip_callback():
    return pl.callbacks.GradientClipCallback(max_norm=1.0, gradient_clip_algorithm='norm')

# Set up the model and trainer
model = Logger()
trainer = pl.Trainer(
    max_epochs=10,
    logger=model.logger,
    callbacks=[GradientPrinter(), gradient_clip_callback()],
    log_every_n_steps=1
)

# Example: Training the model
trainer.fit(model)
