import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time
from torch.utils.data import DataLoader
import sys
import torch.nn.init as init

from shared.embeddings import ConvEmbedding, EnhancedLinearEmbedding, Enhanced__LinearEmbedding, MinimalLinearEmbedding
from shared.self_decode_step import AttentionDecoder
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
        if args['emb_type'] == 'conv':
            self.embedding = ConvEmbedding(prt, args['embedding_dim'])
        elif args['emb_type'] == 'linear':
            self.embedding = MinimalLinearEmbedding(prt, 2, args['embedding_dim'])
        elif args['emb_type'] == 'enhanced':
            self.embedding = EnhancedLinearEmbedding(prt, 2, args['embedding_dim'])
        elif args['emb_type'] == 'enhanced2':
            self.embedding = Enhanced__LinearEmbedding(prt, 2, args['embedding_dim'])

        self.actor_attention = Attention(args['hidden_dim'])

        # Initialize the self-attention based decoder
        self.decodeStep = AttentionDecoder(self.actor_attention, 
                                           num_actions=5, 
                                           args=args)

        self.actor = nn.Sequential(
            self.embedding,
            nn.Linear(args['embedding_dim'], args['hidden_dim']),
            self.actor_attention
        )

        self.critic = Critic(args, 2) #args and input_dim
        self.decoder_input = nn.Parameter(torch.randn(1, 1, args['embedding_dim']))
        init.xavier_uniform_(self.decoder_input)

        # Define optimizers for actor and critic
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.args['actor_net_lr'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.args['critic_net_lr'])

        self.prt.print_out("Agent created - Self Attention.")

    def build_model(self, eval_type= "greedy", show=False, data=None): #prev -> forward
        args = self.args
        batch_size = args['batch_size']

        # if not isinstance(self.env.input_pnt, torch.Tensor):
        #     self.env.input_pnt = torch.tensor(self.env.input_pnt, dtype=torch.float)

        # encoder_emb_inp/context: [batch_size x seq_len x embedding_dim]
        if data is not None:
            input_d = data
            # print('Obtained data not from ENV')
        else:
            if not isinstance(self.env.input_pnt, torch.Tensor):
                self.env.input_pnt = torch.tensor(self.env.input_pnt, dtype=torch.float)
            input_d = self.env.input_pnt
        context = self.embedding(input_d)

        # Reset the Environment.
        self.env.reset()

        # Create tensors and lists
        actions = []
        log_probs = []
        probs = []
        idxs = []

        BatchSequence = torch.arange(batch_size, dtype=torch.int64).unsqueeze(1)
        
        #Decoding loop
        for step in range(args['decode_len']):
            # Get logit and attention weights
            if step == 0:
                # Start from trainable nodes in TSP
                decoder_input = self.decoder_input.expand(batch_size, 1, -1)
            else:
                # Subsequent inputs come from the context based on previous action
                decoder_input = context[batched_idx[:, 0], batched_idx[:, 1]].unsqueeze(1)
            # print('Decoder input: ', decoder_input.size())
            # print('Context: ', context.size())

            action_logits, attn_weights = self.decodeStep(decoder_input, context, self.env.mask)
            # It does softmax inside here
            prob, action_selected = self.decodeStep.select_action(action_logits, eval_type)   

            state = self.env.step(action_selected)

            batched_idx = torch.cat([BatchSequence, action_selected], dim=1).long()
            
            # Advanced indexing in PyTorch to replace gather_nd
            selected_probs = prob[batched_idx[:, 0], batched_idx[:, 1]] 

            # Taking logarithm of the selected probabilities
            logprob = torch.log(selected_probs)

            probs.append(prob)
            idxs.append(action_selected)
            log_probs.append(logprob)

            # Gather using the constructed indices
            action = input_d[batched_idx[:, 0], batched_idx[:, 1]]
            actions.append(action)

        idxs = torch.stack(idxs, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        probs = torch.stack(probs, dim=1)
        actions = torch.stack(actions, dim=1)
        # print(actions.size())
        R = self.reward_func(actions, show=show)

        #Critic part --------------------------------
        v = torch.tensor(0)

        if eval_type == "stochastic":
            # Ensure action_selected is of the correct shape
            action4critic = action_selected.unsqueeze(0).expand(1, batch_size, self.args['hidden_dim']).float()

            # Initialize hidden and cell states for LSTM
            hidden_state = torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim'])
            cell_state = torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim'])
            
            # Forward pass through LSTM
            lstm_layer = nn.LSTM(input_size=self.args['hidden_dim'], hidden_size=self.args['hidden_dim'], num_layers=self.args['rnn_layers'])
            output, (hn, cn) = lstm_layer(action4critic, (hidden_state, cell_state))
            
            # Extract the last hidden state for further processing
            hy = hn[-1]

            for i in range(self.args['n_process_blocks']):
                hy = self.critic(hy, context[torch.arange(batch_size), idxs[-1], :])
            
            v = self.critic.final_step_critic(hy)
            # print('R build model: ', R)

        return (R, v, log_probs, actions, idxs, input_d, probs)



    def build_train_step(self):
        """
        This function returns a train_step op, in which by running it we proceed one training step.
        """

        R, v, log_probs, actions, idxs , batch , probs = self.build_model(eval_type='stochastic') 

        R = R.float()
        v = v.float()
        # print(f"R: {R.mean().item()}")
        # print(f"Build train step - Training R: {R.mean().item()}, v: {v.mean().item()}")

        v_nograd = v.detach()
        # print('v_nograd: ', v_nograd)
        # Losses
        # Advantage should represent the difference between the observed reward and the predicted value
        advantage = R - v_nograd
        # print('Advantage: ', advantage)
        #Size [batch_size]
        logprob_sum = torch.sum(log_probs, dim=1)
        # print('Logprob_sum: ', logprob_sum)
        actor_loss_elements = advantage * logprob_sum
        # Computes actor loss and critic loss
        actor_loss = torch.mean(actor_loss_elements)
        # actor_loss = torch.mean((R - v_nograd) * torch.sum(torch.stack(log_probs), dim=0))
        critic_loss = F.mse_loss(R, v)
        
        # Add monitoring prints
        # print(f"Advantage: {advantage.mean().item()}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")
        # print(f"R: {R.mean().item()}, v: {v.mean().item()}, Advantage: {advantage.mean().item()}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")

        # Compute gradients
        # Clear previous gradients
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # Compute gradients
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
 
        # # Clip gradients (optional, if args['max_grad_norm'] is set)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args['max_grad_norm'])

        # Apply gradients
        self.actor_optim.step()
        self.critic_optim.step()
        
        train_step = [actor_loss.item(), critic_loss.item(), R, v]
        
        return train_step

    def evaluate_single(self, eval_type='greedy'):

        start_time = time.time()
        avg_reward = []

        self.dataGen.reset()
        test_df = self.dataGen.get_test_data()

        test_loader = DataLoader(test_df, batch_size=self.args['batch_size']) 

        for problem_count in range(self.dataGen.n_problems):
            for data in test_loader:                
                self.env.input_data = data

                R, v, log_probs, actions, idxs, batch, _ = self.build_model(eval_type)

                avg_reward.append(R)
                R_ind0 = 0

                # Sample Decode
                if problem_count % int(self.args['log_interval']) == 0:
                    example_output = []
                    example_input = []
                    
                    for i in range(self.env.n_nodes):
                        example_input.append(list(batch[0, i, :]))
                    
                    for idx, action in enumerate(actions):
                        example_output.append(list(action[R_ind0*np.shape(batch)[0]]))
                    
                    # self.prt.print_out('\n\nVal-Step of {}: {}'.format(eval_type, problem_count))
                    # self.prt.print_out('\nExample test input: {}'.format(example_input))
                    # self.prt.print_out('\nExample test output: {}'.format(example_output))
                    self.prt.print_out('\nExample test reward: {} - best: {}'.format(R[0],R_ind0))
                
        end_time = time.time() - start_time
                
        # Finished going through the iterator dataset.
        self.prt.print_out('\n Validation overall avg_reward: {}'.format(np.mean(avg_reward)) )
        self.prt.print_out('Validation overall reward std: {}'.format(np.sqrt(np.var(avg_reward))) )

        self.prt.print_out("Finished evaluation with %d steps in %s." % (problem_count\
                           ,time.strftime("%H:%M:%S", time.gmtime(end_time))))
        
    
    def evaluate_batch(self, eval_type='greedy'):
        
        self.env.reset()
        total_reward = []
        self.dataGen.reset()
        test_df = self.dataGen.get_test_data()
        test_loader = DataLoader(test_df, batch_size=self.args['batch_size'])

        for data in test_loader:
            if data.size(0) != self.args['batch_size']:
                break
            
            R, v, log_probs, actions, idxs, batch, _ = self.evaluate_model(data, eval_type)
            # print('EB - R mean: ', R.mean().item())
            total_reward.extend(R.tolist())

        # if np.array_equal(self.env.input_data, data):
        #     self.prt.print_out("The data is the same.!!!!!!")
        #     sys.exit()
        # self.env.input_data = data

        avg_reward = np.mean(total_reward)
        std_reward = np.std(total_reward)
        self.prt.print_out(f'Evaluate batch - Average of {eval_type} in batch-mode: {avg_reward} -- std R: {std_reward}')

    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
        elif infer_type == 'single':
            self.evaluate_single('single')

        self.prt.print_out("##################################################################")

    def run_train_step(self):
        data = self.dataGen.get_train_next()
        self.env.input_data = data
        train_results = self.build_train_step()
        return train_results

    def evaluate_model(self, data, eval_type='greedy'):
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
        # start_time = time.time()

        for data in test_loader:
            # print('Data shape: ', data.size())
            if data.size(0) != self.args['batch_size']:
                # Fix this! as we are not testing the total amount of data
                # Remember to fix also in attention
                break
            # self.env.input_data = data  # Set the environment's input data to the batch provided by DataLoader
    
            # Run model evaluation for the current batch
            R, v, log_probs, actions, idxs, batch, _ = self.build_model(eval_type, show=True, data=data)
            # print('EM - Actions: ', actions)
            # print('EM - Rewards mean: ', R.mean().item()) # -> A lot of rewards are 0.0
            total_reward.extend(R.tolist())  # Append rewards for 'greedy' or other single-path evaluations
        
        avg_reward = np.mean(total_reward)
        std_reward = np.std(total_reward)
        self.prt.print_out(f'Evaluate model: Average Reward: {avg_reward}, Reward Std Dev: {std_reward}')

        return R, v, log_probs, actions, idxs, batch, _


class Critic(nn.Module):
    def __init__(self, args, input_dim):
        super(Critic, self).__init__()
        self.args = args
        self.attention = Attention(args['hidden_dim'], use_tanh=args['use_tanh'], C=args['tanh_exploration']) 
        self.linear1 = nn.Linear(args['hidden_dim'], args['hidden_dim'])
        self.linear2 = nn.Linear(args['hidden_dim'], 1)

    def forward(self, hidden_state, encoder_outputs):
        e, logit = self.attention(hidden_state, encoder_outputs)
        # print('logits critic: ', logit)
        # print('e critic: ', e.size())
        prob = F.softmax(logit, dim=-1)
        # print('prob critic: ', prob)
        prob_expanded = prob.unsqueeze(1)
        result = torch.matmul(prob_expanded, e) 
        hy = result.squeeze(1)
        return hy
    
    def final_step_critic(self, hy):
        x = F.relu(self.linear1(hy))
        x = self.linear2(x)
        v = x.squeeze(1)

        return v
    
  