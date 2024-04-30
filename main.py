import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import sys

# Import libraries from the task (TSP)
from TSP.tsp_datagen import DataGenerator
from TSP.tsp_env import VRPEnvironment, reward_func
from shared.attention import Attention
from model.attention_agent import RLAgent
from configs import ParseParams
from shared.model_manager import ModelManager

# Problems:
# 1. The model isn't being saved, therefore the variables arent being reused in the correct way (I think)
# 5. We have to save the model and print the results
# 6. Evaluate the model to see performance
# 7. See how to run on cloud - LIACS servers
# 8. See tests for the model on Readme, test all parameters

class principal(nn.Module):
    def __init__(self, args, prt):
        super(principal, self).__init__()
        # Load task specific classes
        self.dataGen = DataGenerator(args)
        self.dataGen.reset()

        self.env = VRPEnvironment(args)
        
        # If task TSP
        self.AttentionActor = Attention
        self.AttentionCritic = Attention

        # create an RL Agent (Network)
        self.agent = RLAgent(args,
                        prt,
                        self.env,
                        self.dataGen,
                        reward_func,
                        self.AttentionActor,
                        self.AttentionCritic,
                        is_train=args['is_train']) # Model class
        
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)

        # self.manager = ModelManager(self.agent, self.optimizer, args)
        # self.manager.initialize()

        # train or evaluate the agent
        start_time = time.time()

        if args['is_train']:
            print("Training started ...")
            self.train()

        else: # inference/ evaluation
            prt.print_out('Evaluation started ...')
            self.agent.inference(args['infer_type'])

        prt.print_out(f'Total time is {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    def train(self):

        prt.print_out('Training the agent...')
        # print("Amount of training steps: ", args['n_train'])

        train_time_beg = time.time()
        
        for step in range(args['n_train']):

            actor_loss_val, critic_loss_val, R_val, v_val = self.agent.run_train_step()
            
            # if step%args['save_interval'] == 0:
                # ModelManager.save_model(self.agent, self.optimizer, step+1, args['model_dir'])

            if step%args['log_interval'] == 0:    

                train_time_end = time.time()-train_time_beg

                prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'.format(step,time.strftime("%H:%M:%S", time.gmtime(train_time_end)), R_val.mean().item(), v_val.mean().item()))
                prt.print_out('    actor loss: {} -- critic loss: {}'.format(np.mean(actor_loss_val),np.mean(critic_loss_val)))
                
                train_time_beg = time.time()

            if step%args['test_interval'] == 0:
                self.agent.inference(args['infer_type'])

if __name__ == "__main__":
    args, prt = ParseParams()

    if args['random_seed'] is not None and args['random_seed'] > 0:

        prt.print_out(f"# Set random seed to {args['random_seed']}")
        np.random.seed(args['random_seed'])
        torch.manual_seed(args['random_seed'])
        
    run_code = principal(args, prt)  # Create an instance