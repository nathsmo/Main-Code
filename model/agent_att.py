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
from shared.self_decode_step import AttentionDecoder as SelfAttention
from shared.decode_step import DecodeStep as PointerNetwork
from shared.attention import Attention

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
        
        if args['decoder'] == "pointer":
            self.decodeStep = PointerNetwork(self.actor_attention, args['hidden_dim'],
                                            n_glimpses=args['n_glimpses'],
                                            mask_glimpses=args['mask_glimpses'],
                                            mask_pointer=args['mask_pointer'],
                                            rnn_layers=args['rnn_layers'])        
        elif args['decoder']=='self':
            # Initialize the self-attention based decoder
            self.decodeStep = SelfAttention(self.actor_attention, 
                                            num_actions=self.env.n_nodes, 
                                           args=args)
        else:
            raise ValueError("Invalid decoder type specified. Supported types: pointer, self_attention, fast.")

        self.actor = nn.Sequential(
            self.embedding,
            nn.Linear(args['embedding_dim'], args['hidden_dim']),
            self.actor_attention
        )

        self.critic = Critic(args, 2) #args and input_dim
        self.decoder_input = nn.Parameter(torch.randn(1, 1, args['embedding_dim']))
        init.xavier_uniform_(self.decoder_input)

        # Define optimizers for actor and critic
        # self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.args['actor_net_lr'])
        # self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.args['critic_net_lr'])
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
        """
        Constructs the computation graph (or computational steps) for the model. 
        Initializes and prepares all necessary inputs, embeddings, and settings for 
        the batch processing of data through the neural network, particularly 
        focusing on the decoding or inference of the next actions (or steps) given 
        the current environment's state.

        Parameters:
        env.n_nodes are the tsp nodes = 10, 20...

        Returns:
        R: The reward for the current batch of data.
        v: The value of the critic network.
        log_probs: The log probabilities of the actions taken.
        actions: The actions taken by the agent.
        idxs (list): The indices of the actions taken. (tsp, batch_size)
            idx might look like [5, 2, 6, ..., 1] (with values for all 128 instances).
        """
        args = self.args
        batch_size = args['batch_size']

        if data is not None:
            input_d = data
        else:
            if not isinstance(self.env.input_pnt, torch.Tensor):
                self.env.input_pnt = torch.tensor(self.env.input_pnt, dtype=torch.float)
            input_d = self.env.input_pnt
            
        if torch.isnan(input_d).any() or torch.isinf(input_d).any():
            print("Input data contains NaNs or Infs")
            sys.exit()

        context = self.embedding(input_d)
        self._check_for_nan()

        # Reset the Environment.
        self.env.reset()
        # Create tensors and lists
        actions, log_probs, probs, idxs = [], [], [], []
        # This should be a list ranging from 0 to batch_size, size [batch_size, 1]
        BatchSequence = torch.arange(batch_size, dtype=torch.int64).unsqueeze(1)
        visited_positions = torch.zeros((batch_size, 1), dtype=torch.long, device=input_d.device)

        if self.args['decoder'] == "pointer":
            initial_state = self._init_hidden(batch_size)
            states = [initial_state]

        for step in range(1, args['decode_len']):
            if step == 1:
                decoder_input = self.decoder_input.expand(batch_size, 1, -1)
            else:
                decoder_input = context[batched_idx[:, 0], batched_idx[:, 1]].unsqueeze(1)

            if self.args['decoder'] == "pointer":
                action_logits, new_state = self.decodeStep(decoder_input, context, self.env.mask, states[-1]) #self.env.mask
                states.append(new_state)
            elif self.args['decoder'] == "self":
                action_logits, attn_weights = self.decodeStep(decoder_input, context, self.env.mask) #self.env.mask

            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                print("Problem with action logits.")
                print("Step: ", step)
                print("Decoder input: ", decoder_input)
                print("Before embeddings input_d: ", input_d)
                print("context: ", context)
                print("Mask env: ", self.env.mask)
                print("Action logits:", action_logits)
                print("eval_type:", eval_type)
                print("First array Visited positions:", visited_positions[0])
                print("Last array Visited positions:", visited_positions[-1])

                raise ValueError("masked_prob contains NaN or Inf values")
            
            prob, action_selected = self.select_action(action_logits, eval_type, visited_positions)

            batched_idx = torch.cat([BatchSequence, action_selected], dim=1).long()
            selected_probs = prob[batched_idx[:, 0], batched_idx[:, 1]]
            logprob = torch.log(selected_probs)

            probs.append(prob)
            idxs.append(action_selected)
            log_probs.append(logprob)
            action = input_d[batched_idx[:, 0], batched_idx[:, 1]]
            actions.append(action)
            visited_positions = torch.cat([visited_positions, action_selected], dim=1)

        idxs = torch.stack(idxs, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        probs = torch.stack(probs, dim=1)
        actions = torch.stack(actions, dim=1)

        R = self.reward_func(actions, show=show)

        # Critic part --------------------------------
        v = torch.tensor(0)

        if eval_type == "stochastic":
            action4critic = action_selected.unsqueeze(0).expand(1, batch_size, self.args['hidden_dim']).float()
            lstm_layer = nn.LSTM(input_size=self.args['hidden_dim'], hidden_size=self.args['hidden_dim'], num_layers=self.args['rnn_layers'])
            output, (hn, cn) = lstm_layer(action4critic, self._init_hidden(batch_size))
            hy = hn[-1]

            for i in range(self.args['n_process_blocks']):
                hy = self.critic(hy, context[torch.arange(batch_size), idxs[-1], :])

            v = self.critic.final_step_critic(hy)

        return (R, v, log_probs, actions, idxs, input_d, probs)

    def build_train_step(self):
        R, v, log_probs, actions, idxs, batch, probs = self.build_model("stochastic")

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

        # for name, param in self.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} gradient: {param.grad.norm()}")

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.get('max_grad_norm', 1.0))
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.get('max_grad_norm', 1.0))

        self.actor_optim.step()
        self.critic_optim.step()

        self._check_for_nan()  # Check for NaNs after optimizer step

        train_step = [actor_loss.item(), critic_loss.item(), R, v]

        return train_step

    # def build_train_step(self):
    #     """
    #     This function returns a train_step operation, in which by running it we proceed one training step.
    #     """
    #     R, v, log_probs, actions, idxs, batch, probs = self.build_model("stochastic")

    #     # Convert R and v to float tensors
    #     R = R.float()
    #     v = v.float()

    #     # In Detach function we separate a tensor from the computational graph by returning a new tensor that doesn't require a gradient
    #     v_nograd = v.detach()

    #     # Size [batch_size]
    #     advantage = R - v_nograd
    #     #Size [batch_size]
    #     logprob_sum = torch.sum(log_probs, dim=1)

    #     actor_loss_elements = advantage * logprob_sum
    #     # Computes actor loss and critic loss
    #     actor_loss = torch.mean(actor_loss_elements)
    #     critic_loss = F.mse_loss(R, v)
    
    #     # Combine losses for a single backward pass
    #     total_loss = actor_loss + critic_loss

    #     # Compute gradients and update parameters
    #     self.actor_optim.zero_grad()
    #     self.critic_optim.zero_grad()
    #     total_loss.backward()

    #     # Compute gradients
    #     # actor_loss.backward(retain_graph=True)  # Retain graph to compute gradients for critic 
    #     # critic_loss.backward(retain_graph=True) #See if this affects anything

    #     # # Clip gradients (optional, if args['max_grad_norm'] is set)
    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args['max_grad_norm'])
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args['max_grad_norm'])
    #     torch.nn.utils.clip_grad_norm_(self.embedding.parameters(), max_norm=1.0)

    #     # Apply gradients
    #     self.actor_optim.step()
    #     self.critic_optim.step()

    #     self._check_for_nan()  # Check for NaNs after optimizer step
        
    #     train_step = [actor_loss.item(), critic_loss.item(), R, v]
        
    #     return train_step

    def evaluate_single(self, eval_type="greedy"):

        start_time = time.time()
        avg_reward = []

        self.dataGen.reset()
        test_df = self.dataGen.get_test_data()

        test_loader = DataLoader(test_df, batch_size=self.args['batch_size']) 

        for problem_count in range(self.dataGen.n_problems):
            for data in test_loader:                
                self.env.input_data = data

                R, v, log_probs, actions, idxs, batch, _ = self.build_model(eval_type)

                if eval_type == 'greedy':
                    avg_reward.append(R)
                    R_ind0 = 0

                # Sample Decode
                if problem_count % int(self.args['log_interval']) == 0:
                    example_output = []
                    example_input = []
                    
                    for i in range(self.env.n_nodes):
                        example_input.append(list(batch[0, i, :]))
                    
                    for idx, action in enumerate(actions):
                        here = list(action[R_ind0*np.shape(batch)[0]])
                        example_output.append(list(action[R_ind0*np.shape(batch)[0]]))
                    
                    # self.prt.print_out('\n\nVal-Step of {}: {}'.format(eval_type, problem_count))
                    # self.prt.print_out('\nExample test input: {}'.format(example_input))
                    # self.prt.print_out('\nExample test output: {}'.format(example_output))
                    self.prt.print_out('\nExample test reward: {} - best: {}'.format(R[0], R_ind0))
                
        end_time = time.time() - start_time
        
        # Finished going through the iterator dataset.
        self.prt.print_out('\n Validation overall avg_reward: {}'.format(np.mean(avg_reward)) )
        self.prt.print_out('Validation overall reward std: {}'.format(np.sqrt(np.var(avg_reward))) )

        self.prt.print_out("Finished evaluation with %d steps in %s." % (problem_count\
                           ,time.strftime("%H:%M:%S", time.gmtime(end_time))))

        if self.args['print_route']:
            print("Calculated Route:")
            for step, action in enumerate(actions):
                print(f"Step {step}: Node at coordinates {action.numpy()}")

    def evaluate_batch(self, eval_type='greedy'):
        
        self.env.reset()
        self.dataGen.reset()
        data = self.dataGen.get_test_data()

        if self.args['decoder'] == "pointer":
            self.env.input_data = data
            R, v, log_probs, actions, idxs, batch, _ = self.evaluate_model(eval_type, data, show=False)
            std_r = R.std().item()
            avg_r = R.mean().numpy()

        else:
            total_reward = []
            test_loader = DataLoader(data, batch_size=self.args['batch_size'])

            for data in test_loader:
                if data.size(0) != self.args['batch_size']:
                    break
                R, v, log_probs, actions, idxs, batch, _ = self.evaluate_model(eval_type, data, show=False)
                total_reward.extend(R.tolist())

            avg_r = np.mean(total_reward)
            std_r = np.std(total_reward)
        self.prt.print_out('Average of {} in batch-mode: {} -- std R: {}'.format(eval_type, avg_r, str(std_r)))  

    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
        elif infer_type == 'single':
            self.evaluate_single('single')

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
                R, v, log_probs, actions, idxs, batch, _ = self.build_model(eval_type, show)
            else:
                R, v, log_probs, actions, idxs, batch, _ = self.build_model(eval_type, show=True, data=data)
            
            # print('EM - Actions: ', actions)
            # print('EM - Rewards mean: ', R.mean().item()) # -> A lot of rewards are 0.0
            total_reward.extend(R.tolist())  # Append rewards for 'greedy' or other single-path evaluations
        
        return R, v, log_probs, actions, idxs, batch, _
    
    def select_action(self, action_logits, method='greedy', visited_positions=None):
        # Clip action_logits to avoid extremely large values
        action_logits = torch.clamp(action_logits, -10, 10)

        # Check for NaNs or Infs in action_logits
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            print("NaNs or Infs detected in action_logits before softmax")
            print("action_logits:", action_logits)
            raise ValueError("NaNs or Infs detected in action_logits before softmax")

        prob = F.softmax(action_logits, dim=-1)

        if visited_positions is not None:
            mask = torch.ones_like(prob)
            mask.scatter_(1, visited_positions, 0)
            masked_prob = prob * mask
            masked_prob = masked_prob / masked_prob.sum(dim=1, keepdim=True)
            if torch.isnan(masked_prob).any() or torch.isinf(masked_prob).any():
                print("Masked probabilities sum to zero, which means all actions are masked out. This should not happen.")
                print("Action logits:", action_logits)
                print("Probabilities:", prob)
                print("Masked probabilities:", masked_prob)
                print("First array Visited positions:", visited_positions[0])
                print("Last array Visited positions:", visited_positions[-1])
                raise ValueError("masked_prob contains NaN or Inf values")

        if method == "greedy":
            random_values = torch.rand(masked_prob.size(0), 1, device=masked_prob.device)
            greedy_idx = torch.argmax(masked_prob, dim=1, keepdim=True)
            idx = torch.where(
                random_values < self.args['epsilon'],
                torch.randint(masked_prob.size(1), (masked_prob.size(0), 1), device=masked_prob.device),  # Random action
                greedy_idx  # Greedy action
            )
            self.args['epsilon'] *= 0.9999
        elif method == "stochastic":
            idx = torch.multinomial(masked_prob, num_samples=1, replacement=True)
        
        return masked_prob, idx

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim']),
                torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim']))
    
    def process_decoder_step_per_instance(self, context, mask=None):
        # x = context.long()
        batch_size = context.size(0)
        action_logits_list = []
        mean_tensor = torch.mean(context, dim=-1)

        # Map the mean values to integers, you can use a simple linear transformation or quantization
        # Here's a basic example using a linear transformation to map to integer range [0, N)
        N = 4096  # Example integer range, change according to your requirements
        integer_tensor = (mean_tensor - mean_tensor.min()) / (mean_tensor.max() - mean_tensor.min())  # Normalize to [0, 1]
        integer_tensor = (integer_tensor * N).long()  # Scale to [0, N) and convert to long

        for i in range(batch_size):
            instance_context = integer_tensor[i].unsqueeze(0)  # Extract a single instance and add batch dimension
            print('instance_context', instance_context)
            if mask is not None:
                instance_mask = mask[i].unsqueeze(0)
                instance_action_logits = self.decodeStep(instance_context, mask=instance_mask)
            else:
                instance_action_logits = self.decodeStep(instance_context)
        
            action_logits_list.append(instance_action_logits)
        
        # Combine the action logits into a single tensor
        combined_action_logits = torch.cat(action_logits_list, dim=0)
        
        return combined_action_logits
    
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
    
    
