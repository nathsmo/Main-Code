import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import sys

# Import libraries from the task (TSP)
from TSP.tsp_datagen import DataGenerator
from TSP.tsp_env import VRPEnvironment, reward_func
from shared.attention import Attention
from model.attention_agent import RLAgent
from model.self_attention_agent import RLAgent as SelfAttentionAgent

from configs import ParseParams

class principal(nn.Module):
    def __init__(self, args, prt):
        super(principal, self).__init__()
        # Load task specific classes
        self.dataGen = DataGenerator(args)
        self.dataGen.reset()

        self.env = VRPEnvironment(args)
        
        self.AttentionActor = Attention
        self.AttentionCritic = Attention

        if args['decoder'] == 'self':
            self.agent = SelfAttentionAgent(args,
                            prt,
                            self.env,
                            self.dataGen,
                            reward_func,
                            self.AttentionActor,
                            self.AttentionCritic,
                            is_train=args['is_train']) # Model class
            
        elif args['decoder'] == 'pointer':
            # create an RL Agent (Network)
            self.agent = RLAgent(args,
                            prt,
                            self.env,
                            self.dataGen,
                            reward_func,
                            self.AttentionActor,
                            self.AttentionCritic,
                            is_train=args['is_train']) # Model class
        else:
            raise Exception('Decoder not implemented')
        
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)

        # Set up TensorBoard SummaryWriter
        self.writer = SummaryWriter()

        # train or evaluate the agent
        start_time = time.time()
        
        if args['is_train']:
            prt.print_out("Training started ...")
            self.train()
            torch.save(self.agent.state_dict(), f"{args['model_dir']}/agent_complete.pth")

        else: # inference/ evaluation
            prt.print_out('Evaluation started ...')
            self.agent.load_state_dict(torch.load(args['model_dir']))
            self.agent.eval()
            self.agent.inference(args['infer_type'])

        prt.print_out(f'Total time is {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    def train(self):
        prt.print_out('Training the agent...')
        train_time_beg = time.time()
        
        for step in range(args['n_train']):
            actor_loss_val, critic_loss_val, R_val, v_val = self.agent.run_train_step()
            # Logging to TensorBoard
            self.writer.add_scalar('Train/Reward', R_val.mean().item(), step)
            self.writer.add_scalar('Train/Value', v_val.mean().item(), step)
            self.writer.add_scalar('Train/Actor_Loss', np.mean(actor_loss_val), step)
            self.writer.add_scalar('Train/Critic_Loss', np.mean(critic_loss_val), step)
            
            if step % args['save_interval'] == 0:
                torch.save(self.agent.state_dict(), f"{args['model_dir']}/model_{step}.pth")

            if step % args['log_interval'] == 0:    
                train_time_end = time.time()-train_time_beg
                prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'.format(step,time.strftime("%H:%M:%S", time.gmtime(train_time_end)), R_val.mean().item(), v_val.mean().item()))
                prt.print_out('    actor loss: {} -- critic loss: {}'.format(np.mean(actor_loss_val),np.mean(critic_loss_val)))
                
                train_time_beg = time.time()

            if step % args['test_interval'] == 0:
                self.agent.inference(args['infer_type'])

        # Close the SummaryWriter
        self.writer.close()

if __name__ == "__main__":
    args, prt = ParseParams()

    if args['random_seed'] is not None and args['random_seed'] > 0:
        prt.print_out(f"# Set random seed to {args['random_seed']}")
        prt.print_out(f"# Set parameters for this run:")
        prt.print_out(f"# Variation:    {args['variation']}")
        prt.print_out(f"# Task:         {args['task']}")
        prt.print_out(f"# Decoder:      {args['decoder']}")
        prt.print_out(f"# Embed type:   {args['emb_type']}")
        prt.print_out(f"# n_glimpses:   {args['n_glimpses']}")
        prt.print_out(f"# rnn layers:   {args['rnn_layers']}")
        prt.print_out(f"# n_train:      {args['n_train']}")
        prt.print_out(f"# num heads:    {args['num_heads']}")

        np.random.seed(args['random_seed'])
        torch.manual_seed(args['random_seed'])

    run_code = principal(args, prt)  # Create an instance
