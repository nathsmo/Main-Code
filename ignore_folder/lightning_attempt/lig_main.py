import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Import necessary modules
from TSP.tsp_datagen import DataGenerator
from TSP.tsp_env import VRPEnvironment, reward_func
from ignore_folder.attention_agent import RLAgent
from ignore_folder.self_attention_agent import RLAgent as SelfAttentionAgent
from configs import ParseParams
from shared.attention import Attention

# Problems:
# * See tests for the model on Readme (test all parameters)


class Principal(pl.LightningModule):
    def __init__(self, args, prt):
        super().__init__()
        self.args = args
        self.prt = prt
        self.dataGen = DataGenerator(args)
        self.env = VRPEnvironment(args)
        self.AttentionActor = Attention
        self.AttentionCritic = Attention

        if args['decoder'] == 'self':
            self.agent = SelfAttentionAgent(args, prt, self.env, self.dataGen, reward_func,
                                            self.AttentionActor, self.AttentionCritic, is_train=args['is_train'])
        else:
            self.agent = RLAgent(args, prt, self.env, self.dataGen, reward_func,
                                 self.AttentionActor, self.AttentionCritic, is_train=args['is_train'])
        
        self.save_hyperparameters()  # To log hyperparameters automatically

    def forward(self, x):
        return self.agent(x)

    def training_step(self, batch, batch_idx):
        actor_loss_val, critic_loss_val, R_val, v_val = self.agent.run_train_step()
        self.log('train_loss', actor_loss_val + critic_loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': actor_loss_val + critic_loss_val, 'reward': R_val.mean().item(), 'value': v_val.mean().item()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        # Create your DataLoader
        return DataLoader(...)  # Adjust accordingly

if __name__ == "__main__":
    args, prt = ParseParams()
    if args['random_seed'] is not None and args['random_seed'] > 0:
        np.random.seed(args['random_seed'])
        torch.manual_seed(args['random_seed'])

    model = Principal(args, prt)
    trainer = pl.Trainer(max_epochs=args['max_epochs'], gpus=args['gpus'])  # Adjust according to your setup
    trainer.fit(model)
