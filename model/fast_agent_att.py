from torch.distributions import Categorical
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import time
import sys

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)

from shared.embeddings import ConvEmbedding, EnhancedLinearEmbedding, Enhanced__LinearEmbedding, MinimalLinearEmbedding
from shared.attention import Attention
from fast_transformer_pytorch import FastTransformer

class RLAgent(nn.Module):
    def __init__(self, args, prt, env, dataGen, reward_func, is_train=True):
        super(RLAgent, self).__init__()
        self.args = args
        self.prt = prt
        self.env = env
        self.dataGen = dataGen
        self.reward_func = reward_func
        # Embedding and Decoder setup
        if self.args['emb_type'] == "conv":
            self.embedding = ConvEmbedding(prt, args['embedding_dim'])
        elif self.args['emb_type'] == "enhanced":
            self.embedding = EnhancedLinearEmbedding(prt, 2, args['embedding_dim'])
        elif self.args['emb_type'] == "enhanced2":
            self.embedding = Enhanced__LinearEmbedding(prt, 2, args['embedding_dim'])
        elif self.args['emb_type'] == "linear":
            self.embedding = MinimalLinearEmbedding(prt, 2, args['embedding_dim'])
        else:
            raise ValueError("Invalid embedding type specified. Supported types: linear, enhanced, minimal")
        
        # Separate attention mechanisms
        self.actor_attention = Attention(args['hidden_dim'])
        
        self.decodeStep = FastTransformer(
            num_tokens=20000,  # This should match the vocabulary size, adjust if necessary
            dim=512,
            depth=2,
            max_seq_len=4096,
            absolute_pos_emb=True
        )
                
        self.actor = nn.Sequential(
            self.embedding,
            nn.Linear(args['embedding_dim'], args['hidden_dim']),
            self.actor_attention
        )

        self.critic = Critic(args, 2) #args and input_dim
        self.decoder_input = nn.Parameter(torch.randn(1, 1, args['embedding_dim']))
        init.xavier_uniform_(self.decoder_input)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.args.get('actor_net_lr', 1e-4))
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.args.get('critic_net_lr', 1e-4))

        self.prt.print_out("Agent created - " + args['decoder'])

    def _initialize_weights(self):
        if hasattr(self.embedding, '_initialize_weights'):
            self.embedding._initialize_weights()

    def _check_for_nan(self):
        for name, param in self.embedding.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"NaNs or Infs detected in {name}")
            
    def build_model(self, eval_type="greedy", show=False, data=None):
        args = self.args
        batch_size = args['batch_size']

        if data is not None:
            input_d = data
        else:
            if not isinstance(self.env.input_pnt, torch.Tensor):
                self.env.input_pnt = torch.tensor(self.env.input_pnt, dtype=torch.float)
            input_d = self.env.input_pnt
            
        # Reset the Environment.
        self.env.reset()
        # Create tensors and lists
        actions, log_probs, probs = [], [], []
        # -----------------------------
        action_logits, index_tensor = self.fastformer_process(input_d)
        # Select actions using greedy method
        selected_coordinate_indices_greedy = self.select_action_greedy(action_logits)
        # print("Selected coordinate indices (greedy):", selected_coordinate_indices_greedy)
        # print("Selected coordinate indices (greedy):", selected_coordinate_indices_greedy.size())
        # Apply the function to each row of the tensor
        unique_selected_coordinate_indices_greedy = torch.stack([self.remove_duplicates_preserve_order(row) for row in selected_coordinate_indices_greedy])
        # print("Selected unique coordinate indices (greedy):", result)
        # print("Selected unique coordinate indices (greedy):", result.size())
        actions = self.obtain_original_coordinates(input_d, unique_selected_coordinate_indices_greedy) #selected_coordinates_greedy
        # print("Selected coordinates (greedy):", selected_coordinates_greedy)
        
        # -----------------------------
        # batched_idx = selected_coordinate_indices_greedy
        prob = F.softmax(action_logits, dim=-1) # I need the action logits only of the selected ones # selected_coordinate_indices_greedy maybe?
        selected_probs = prob[selected_coordinate_indices_greedy[:, 0], selected_coordinate_indices_greedy[:, 1]]
        log_probs = torch.log(selected_probs)
        # log_probs = torch.stack(log_probs, dim=1)
        # probs = torch.stack(probs, dim=1)
        R = self.reward_func(actions, show=show)
        # Critic part --------------------------------
        v = torch.tensor(0)

        if eval_type == "stochastic":
            # action4critic = action_selected.unsqueeze(0).expand(1, batch_size, self.args['hidden_dim']).float()
            # action4critic = actions # Change the action4critic #torch.Size([1, 64, 128])
            # action_selected = last column of actions and then do the action4critic
            last_column = actions[:, -1]  # Shape [64]
            print('Size of last column: ', last_column.size())
            print('Size of actions: ', actions.size())
            # Step 2: Expand the extracted column to the desired shape [1, 64, 128]
            # We can use unsqueeze to add the necessary dimensions and then expand
            action4critic = last_column.unsqueeze(0).float()
            lstm_layer = nn.LSTM(input_size=self.args['hidden_dim'], hidden_size=self.args['hidden_dim'], num_layers=self.args['rnn_layers'])
            output, (hn, cn) = lstm_layer(action4critic, self._init_hidden(2))
            hy = hn[-1]

            for i in range(self.args['n_process_blocks']):
                hy = self.critic(hy, input_d)
            v = self.critic.final_step_critic(hy)

        return (R, v, log_probs, actions, input_d, probs)

    def build_train_step(self):
        R, v, log_probs, actions, batch, probs = self.build_model("stochastic")

        R = R.float()
        v = v.float()

        v_nograd = v.detach()

        advantage = (R - v_nograd).detach()
        logprob_sum = torch.sum(log_probs, dim=1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        actor_loss_elements = advantage * logprob_sum
        actor_loss = torch.mean(actor_loss_elements) - self.args.get('entropy_weight', 0.01) * entropy
        critic_loss = F.mse_loss(R, v)

        total_loss = actor_loss + self.args.get('critic_loss_weight', 1.0) * critic_loss

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.get('max_grad_norm', 1.0))
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.get('max_grad_norm', 1.0))

        self.actor_optim.step()
        self.critic_optim.step()

        self._check_for_nan()  # Check for NaNs after optimizer step
        train_step = [actor_loss.item(), critic_loss.item(), R, v]

        return train_step

    def evaluate_batch(self, eval_type='greedy'):
        
        self.env.reset()
        self.dataGen.reset()
        data = self.dataGen.get_test_data()

        if self.args['decoder'] == "pointer":
            self.env.input_data = data
            R, v, log_probs, actions, batch, _ = self.evaluate_model(eval_type, data, show=False)
            std_r = R.std().item()
            avg_r = R.mean().numpy()

        else:
            total_reward = []
            test_loader = DataLoader(data, batch_size=self.args['batch_size'])

            for data in test_loader:
                if data.size(0) != self.args['batch_size']:
                    break
                R, v, log_probs, actions, batch, _ = self.evaluate_model(eval_type, data, show=False)
                total_reward.extend(R.tolist())

            avg_r = np.mean(total_reward)
            std_r = np.std(total_reward)
        self.prt.print_out('Average of {} in batch-mode: {} -- std R: {}'.format(eval_type, avg_r, str(std_r)))  

    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
        self.prt.print_out("##################################################################")

    def run_train_step(self):
        """
        Obtains the next batch of training data (randomly created) and runs a training step.
        """
        self.env.input_data = self.dataGen.get_train_next()
        train_results = self.build_train_step()
        return train_results

    def evaluate_model(self, eval_type='greedy', data=None, show=False):
        """
        Evaluate the model on the provided DataLoader with a specific evaluation type.
        
        Parameters:
            agent (RLAgent): The agent to evaluate.
            data_loader (DataLoader): DataLoader providing the test dataset.
            eval_type (str): Type of evaluation, e.g., 'greedy', 'beam_search'.
        
        Returns:
            Tuple containing average reward and standard deviation of rewards.
        """

        total_reward = []
        self.dataGen.reset()
        test_df = self.dataGen.get_test_data()
        test_loader = DataLoader(test_df, batch_size=self.args['batch_size']) #, collate_fn=lambda x: padded_collate(x, self.args['batch_size'])) 

        for data in test_loader:
            # print('Data shape: ', data.size())
            if data.size(0) != self.args['batch_size']:
                break # Fix this! as we are not testing the total amount of data

            #Check to see if this has somethign to do with teh pointer network performance
            if self.args['decoder'] == "pointer":
                self.env.input_data = data
    
            # Run model evaluation for the current batch
            if self.args['decoder'] == "pointer":
                R, v, log_probs, actions, batch, _ = self.build_model(eval_type, show)
            else:
                R, v, log_probs, actions, batch, _ = self.build_model(eval_type, show=True, data=data)
            
            # print('EM - Actions: ', actions)
            # print('EM - Rewards mean: ', R.mean().item()) # -> A lot of rewards are 0.0
            total_reward.extend(R.tolist())  # Append rewards for 'greedy' or other single-path evaluations
        
        return R, v, log_probs, actions, batch, _
    
    def fastformer_process(self, original_coordinates):
        # Normalize the coordinates to fit into a range of token sIDs
        # normalized_coordinates = (original_coordinates + 1) / 2  # Now in range [0, 1]
        # original_coordinates should already be in range of [0, 1]
        # Convert normalized coordinates to token IDs
        token_ids_x = torch.clamp((original_coordinates[:, :, 0] * 19999).long(), 0, 19999)
        token_ids_y = torch.clamp((original_coordinates[:, :, 1] * 19999).long(), 0, 19999)
        # Interleave x and y token IDs to form a sequence
        token_ids = torch.stack((token_ids_x, token_ids_y), dim=2).view(64, -1)  # Shape: (64, 20)
        # Pad the token IDs to fit the max_seq_len expected by the transformer
        padded_token_ids = nn.functional.pad(token_ids, (0, 4096 - token_ids.size(1)), 'constant', 0)  # Shape: (64, 4096)
        # Create a mask for the sequence
        mask = torch.zeros(64, 4096).bool()  # Initialize mask with zeros
        mask[:, :20] = 1  # Set the valid positions to 1
        # Get action logits from the transformer
        logits = self.decodeStep(padded_token_ids, mask=mask)  # Shape: (64, 4096, 20000)
        # Extract the logits corresponding to the original 10 positions
        action_logits = logits[:, :20:2, :]  # Shape: (64, 10, 20000)
        # Create a tensor to map the selected indices to the original coordinates
        index_tensor = torch.arange(10).unsqueeze(0).unsqueeze(-1).expand(64, -1, 2).to(original_coordinates.device)
        # Print the shape of action_logits
        return action_logits, index_tensor
    
    def select_action_greedy(self, action_logits):
        """
        Selects actions using greedy method (highest logit value).
        Args:
            action_logits (torch.Tensor): The action logits of shape (batch_size, num_actions, num_tokens).
        Returns:
            selected_indices (torch.Tensor): The indices of selected actions of shape (batch_size, num_actions).
        """
        # Select the action with the highest logit value
        selected_token_indices = torch.argmax(action_logits, dim=-1)
        selected_coordinate_indices = torch.argmax(action_logits, dim=1)
        return selected_coordinate_indices

    def remove_duplicates_preserve_order(self, tensor_row):
        # Create a mask to keep track of unique elements
        seen = set()
        mask = []
        for elem in tensor_row:
            if elem.item() not in seen:
                seen.add(elem.item())
                mask.append(True)
            else:
                mask.append(False)
        return tensor_row[torch.tensor(mask, dtype=torch.bool)]

    def obtain_original_coordinates(self, original_coordinates, unique_coord):
        # Initialize the result tensor
        result = torch.zeros((self.args['batch_size'], self.args['decode_len'], 2), dtype=original_coordinates.dtype)

        # Iterate over each batch to gather the coordinates based on indices in tensor_2
        for i in range(original_coordinates.size(0)):
            result[i] = original_coordinates[i, unique_coord[i]]
        return result
    
    def _init_hidden(self, batch_size):
        return (torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim']),
                torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim']))
    
    
class Critic(nn.Module):
    def __init__(self, args, input_dim):
        super(Critic, self).__init__()
        self.args = args
        self.attention = Attention(args['hidden_dim'], use_tanh=args['use_tanh'], C=args['tanh_exploration']) 
        self.linear1 = nn.Linear(args['hidden_dim'], args['hidden_dim'])
        self.linear2 = nn.Linear(args['hidden_dim'], 1)

    def forward(self, hidden_state, encoder_outputs):
        e, logit = self.attention(hidden_state, encoder_outputs)
        prob = F.softmax(logit, dim=-1)
        prob_expanded = prob.unsqueeze(1)
        result = torch.matmul(prob_expanded, e) 
        hy = result.squeeze(1)
        return hy
    
    def final_step_critic(self, hy):
        x = F.relu(self.linear1(hy))
        x = self.linear2(x)
        v = x.squeeze(1)

        return v
    
    
