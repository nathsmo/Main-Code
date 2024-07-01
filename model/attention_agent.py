import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time
from torch.utils.data import DataLoader
import sys

from shared.embeddings import LinearEmbedding, EnhancedLinearEmbedding, Enhanced__LinearEmbedding, MinimalLinearEmbedding
from shared.decode_step import DecodeStep
from shared.attention import Attention

class RLAgent(nn.Module):

    def __init__(self, args, prt, env, dataGen, reward_func, is_train=True):
        super().__init__()

        self.args = args
        self.prt = prt
        self.env = env
        self.dataGen = dataGen
        self.reward_func = reward_func

        # Embedding and Decoder setup
        if args['emb_type'] == 'linear':
            self.embedding = LinearEmbedding(prt, args['embedding_dim'])
        elif args['emb_type'] == 'minimal':
            self.embedding = MinimalLinearEmbedding(prt, 2, args['embedding_dim'])
        elif args['emb_type'] == 'enhanced':
            self.embedding = EnhancedLinearEmbedding(prt, 2, args['embedding_dim'])
        elif args['emb_type'] == 'enhanced2':
            self.embedding = Enhanced__LinearEmbedding(prt, 2, args['embedding_dim'])

        # Separate attention mechanisms
        self.actor_attention = Attention(args['hidden_dim'])
        # self.critic_attention = Attention(args['hidden_dim'], use_tanh=args['use_tanh'], C=args['tanh_exploration'])

        self.decodeStep = DecodeStep(self.actor_attention, args['hidden_dim'],
                                        n_glimpses=args['n_glimpses'],
                                        mask_glimpses=args['mask_glimpses'],
                                        mask_pointer=args['mask_pointer'],
                                        rnn_layers=args['rnn_layers'])

        """ nn.Sequential container. This container is used to encapsulate a sequence of layers through which the 
        input data will pass in a straightforward, sequential manner. Each layer or module processes the output 
        of the previous one, culminating in an output that reflects the combined operations of all layers in the sequence.

        For the actor a critic definition:
        The input data first gets transformed into an embedding space, then re-mapped linearly to potentially another dimension, 
        and finally processed through an attention mechanism that focuses on the most relevant aspects of the data.
        """

        self.actor = nn.Sequential(
            self.embedding,
            nn.Linear(args['embedding_dim'], args['hidden_dim']),
            self.actor_attention
        )

        # self.critic = nn.Sequential(
        #     self.embedding,
        #     nn.Linear(args['embedding_dim'], args['hidden_dim']),
        #     self.critic_attention
        # )
        self.critic = Critic(args, 2) #args and input_dim

        self.decoder_input = nn.Parameter(torch.randn(1, 1, args['embedding_dim']))
        nn.init.xavier_uniform_(self.decoder_input)

        # Initialize actor and critic networks
        # self.actor = clAttentionActor(self.args['hidden_dim'])
        # self.critic = clAttentionCritic(self.args['hidden_dim'])

        # Define optimizers for actor and critic
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.args['actor_net_lr'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.args['critic_net_lr'])

        self.prt.print_out("Agent created - Pointer Network.")

    def build_model(self, decode_type= "greedy"):
        """
        Could be seen as the forward pass function.

        The build_model function constructs the computation graph (or computational steps) for the model. 
        It initializes and prepares all necessary inputs, embeddings, and settings for the batch processing of 
        data through the neural network, particularly focusing on the decoding or inference of the next actions 
        (or steps) given the current environment's state.

        Parameters:
        env.n_nodes are the tsp nodes = 10, 20 or 50

        Returns:
        R: The reward for the current batch of data.
        v: The value of the critic network.
        log_probs: The log probabilities of the actions taken.
        actions: The actions taken by the agent.
        idxs (list): The indices of the actions taken. (tsp, batch_size)
            idx might look like [5, 2, 6, ..., 1] (with values for all 128 instances).
        """
        # Builds the model
        batch_size = self.env.input_pnt.shape[0]
        # We disabled the beam_width from the original code
        beam_width = 1

        # Reset the Environment.
        self.env.reset(beam_width)

        # self.env.input_pnt: [batch_size x max_time x input_dim=2]
        if not isinstance(self.env.input_pnt, torch.Tensor):
            self.env.input_pnt = torch.tensor(self.env.input_pnt, dtype=torch.float).float()

        # encoder_emb_inp: [batch_size x max_time x embedding_dim]
        context = self.embedding(self.env.input_pnt) # this is a forward pass

        # Code tryout
        # Decoder State -> First: initial_state -> Then -> new_state (in loop)
        initial_state = self.decodeStep._init_hidden(batch_size)
        
        logits, actions = [], []
        states = [initial_state]

        #old code - batchsequence --------------------------------
        # This should be a list ranging from 0 to batch_size, size [batch_size, 1]
        BatchSequence = torch.arange(batch_size, dtype=torch.int64).unsqueeze(1)
        # print('BatchSequence size: ', BatchSequence.size())

        # create tensors and lists
        actions_tmp = []
        logprobs = []
        probs = []
        idxs = []

        # Start from depot
        # This should be a tensor of size [batch_size, 1] with values of (n_nodes - 1)
        chosen_action = (self.env.n_nodes - 1) * torch.ones([batch_size, 1]) # Before called idx
        # print('chosen_action size: ', chosen_action.size())

        #List with selected coodinate, size [batch_size, 1, 2]
        action = self.env.input_pnt[:, self.env.n_nodes - 1].unsqueeze(1)

        #Decoding loop
        for step in range(self.args['decode_len']):
            # Update the decoder input at each step
            # decoder_input: [batch_size, 1, hidden_dim]
            if step == 0:
                # Start from trainable nodes in TSP
                decoder_input = self.decoder_input.expand(batch_size, 1, -1)
            else:
                # Subsequent inputs come from the context based on previous action
                decoder_input = context[torch.arange(batch_size), actions[-1], :]  # Actions indexed from context
                # print('passed')
                # decoder_input = self.decoder_input.repeat(batch_size, 1, 1)
            # print('Decoder input shape', decoder_input.size()) 
            # print('Mask size: ', self.env.mask.size()) # Mask size should be [batch_size, n_nodes]

            # Send to Decode step
            logit, new_state = self.decodeStep(decoder_input, context, self.env.mask, states[-1])

            # States is list of shape (num-states-appended, 2, hidden_states) where hidden_states = [1, 128, 128] 
            states.append(new_state)
            logprob = F.log_softmax(logit, dim=-1)
            probabilities = F.softmax(logit, dim=1)

            #Chosen action size:  torch.Size([batch_size, 1])
            #Change this to epsilon greedy !!!!!!
            if decode_type == "greedy":
                _, chosen_action = probabilities.max(dim=1)
            elif decode_type == "stochastic":
                chosen_action = probabilities.multinomial(num_samples=1)            
            # print('Chosen action: ', chosen_action)
            
            state = self.env.step(chosen_action)

            #Batched index: torch.Size([batch_size, 2])
            batched_idx = torch.cat([BatchSequence, chosen_action], dim=1).long()

            actions.append(chosen_action) #replaces idxs
            logits.append(logit)
            logprobs.append(logprob)
            probs.append(probabilities)

            # Given action:  torch.Size([batch_size, 2]) (For batch size, coordinates obtained.)
            given_action = self.env.input_pnt[batched_idx[:, 0], batched_idx[:, 1]] # Before called action
            actions_tmp.append(given_action)
            # print('Given action: ', given_action.shape)

        logits = torch.stack(logits, dim=1)
        actions = torch.stack(actions, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        probs = torch.stack(probs, dim=1)
        actions_tmp = torch.stack(actions_tmp, dim=1) # on past code named actions

        R = self.reward_func(actions_tmp)
        # print('Reward: ', R)
        # return logits, actions, states

        #Critic part
        v = torch.tensor(0)
        # print('Chosen action size: ', chosen_action.size())
        #action4critic size = [1, batch_size, hidden_dim]
        action4critic = chosen_action.unsqueeze(0).expand(1, batch_size, self.args['hidden_dim']).float()
        # print(action4critic.shape)
        # Initialize hidden and cell states
        hidden_state = torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim'])
        cell_state = torch.zeros(self.args['rnn_layers'], batch_size, self.args['hidden_dim'])
        # Create the LSTM state tuple
        lstm_state = (hidden_state, cell_state)
        # Use the lstm_state in an LSTM
        lstm_layer = nn.LSTM(input_size=self.args['hidden_dim'], hidden_size=self.args['hidden_dim'], num_layers=self.args['rnn_layers'])
        # Forward pass through LSTM
        output, (hn, cn) = lstm_layer(action4critic, lstm_state)
        # Extracting hy (which in this context would be the last cell state from the first layer)
        hy = cn[0]

        for i in range(self.args['n_process_blocks']):
            hy = self.critic(hy, context[torch.arange(batch_size), actions[-1], :])

        v = self.critic.final_step_critic(hy)
        # print('V: ', v)
        # print("R: ", R)
        return (R, v, logprobs, actions_tmp, actions, self.env.input_pnt, probs)


    def build_train_step(self):
        """
        This function returns a train_step operation, in which by running it we proceed one training step.
        """

        R, v, log_probs, actions, idxs , batch , probs = self.build_model("stochastic") # self.train_model
        # Convert R and v to float tensors
        R = R.float()
        v = v.float()

        # In Detach function we separate a tensor from the computational graph by returning a new tensor that doesn't require a gradient
        v_nograd = v.detach()
        R = R.detach()
        # Size [batch_size]
        advantage = R - v_nograd
        #Size [batch_size, n_nodes]
        logprob_sum = torch.sum(log_probs, dim=1)

        # print('idxs size', idxs.size())
        # print('actions size', actions.size())
        # print('log_probs size', log_probs.size())
        # print('logprob_sum size', logprob_sum.size())
        # print('advantage size', advantage.size())

        actor_loss_elements = advantage.unsqueeze(1) * logprob_sum
        # Computes actor loss and critic loss
        actor_loss = torch.mean(actor_loss_elements) # Changed to R and v from R_nograd and v_nograd
        critic_loss = F.mse_loss(R, v)
    
        # Compute gradients
        # Clear previous gradients
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # Compute gradients
        actor_loss.backward(retain_graph=True)  # Retain graph to compute gradients for critic 
        critic_loss.backward()

        # # Clip gradients (optional, if args['max_grad_norm'] is set)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args['max_grad_norm'])

        # Apply gradients
        self.actor_optim.step()
        self.critic_optim.step()
        
        train_step = [actor_loss.item(), critic_loss.item(), R, v]
        
        return train_step

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

                elif eval_type == 'beam_search':
                    # Assuming R is a PyTorch tensor initially; if R is a NumPy array, convert it using torch.from_numpy(R)
                    # First, expand the dimensions of R at axis 1
                    R = R.unsqueeze(1)
                    # Split R into 'self.args['beam_width']' parts along the first dimension (axis 0)
                    split_R = torch.split(R, split_size_or_sections=int(R.size(0) / self.args['beam_width']), dim=0)
                    # Concatenate the split tensors along the second dimension (axis 1)
                    R = torch.cat(split_R, dim=1)
                    R_val = np.amin(R,1, keepdims = False)
                    R_ind0 = np.argmin(R,1)[0]
                    avg_reward.append(R_val)

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
        if eval_type == 'greedy':
            beam_width = 1
        elif eval_type == 'beam_search':
            beam_width = self.args['beam_width']

        data = self.dataGen.get_test_data()
        start_time = time.time()

        if np.array_equal(self.env.input_data, data):
            self.prt.print_out("The data is the same.!!!!!!")
            sys.exit()

        self.env.input_data = data

        R, v, log_probs, actions, idxs, batch, _ = self.evaluate_model(eval_type)

        if len(R.size()) == 0:
            self.prt.print_out("This is the std of R: ", R.std())
            self.prt.print_out("  R is empty !")
            sys.exit()

        if beam_width > 1:
            # Assuming R is a PyTorch tensor and not a numpy array; if it's a numpy array, convert it first
            # R = torch.tensor(R) if it's initially a numpy array
            R = torch.unsqueeze(R, 1)  # Add a dimension at axis 1

            # Split and then concatenate across the desired dimension
            # Split the tensor into 'beam_width' parts along the new dimension (axis 0, as it becomes after unsqueeze)
            split_tensors = torch.split(R, split_size_or_sections=int(R.shape[0] / beam_width), dim=0)
            R = torch.cat(split_tensors, dim=1)

            # Compute the minimum across the concatenated dimension
            R = torch.min(R, dim=1).values  # torch.min returns a namedtuple (values, indices)


        std_r = R.std().item()

        end_time = time.time() - start_time
        self.prt.print_out('Average of {} in batch-mode: {} -- std R: {} -- time {} s'.format(eval_type, R.mean().numpy(), str(std_r), end_time))  
        
    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
            # self.evaluate_batch('beam_search')
        elif infer_type == 'single':
            self.evaluate_single('single')
            # self.evaluate_single('beam_search')
        
        self.prt.print_out("##################################################################")

    def run_train_step(self):
        """
        Obtains the next batch of training data (randomly created) and runs a training step.
        """
        # Obtains the next batch of training data
        self.env.input_data = self.dataGen.get_train_next()

        train_results = self.build_train_step()
        return train_results

    def evaluate_model(self, eval_type='greedy'):
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
        start_time = time.time()

        for data in test_loader:
            # print('Data shape: ', data.size())
            if data.size(0) != self.args['batch_size']:
                # Fix this! as we are not testing the total amount of data
                # Remember to fix also in self attention
                break
            self.env.input_data = data  # Set the environment's input data to the batch provided by DataLoader
            
            # Run model evaluation for the current batch
            R, v, log_probs, actions, idxs, batch, _ = self.build_model(eval_type)

            if eval_type == 'beam_search':
                # For beam search, handling multiple paths per instance
                R = R.view(-1, self.args['beam_width'])  # Reshape R assuming it is flat with all paths
                R_val, _ = torch.min(R, dim=1)  # Find the minimum reward across beams
                total_reward.extend(R_val.tolist())  # Append to total rewards list
            else:
                total_reward.extend(R.tolist())  # Append rewards for 'greedy' or other single-path evaluations
        end_time = time.time() - start_time

        # self.prt.print_out(f'Average Reward: {R.mean().numpy()}, Reward Std Dev: {R.std().item()}, -- time {end_time} s')

        return R, v, log_probs, actions, idxs, batch, _

class Critic(nn.Module):
    def __init__(self, args, input_dim):
        super(Critic, self).__init__()
        self.args = args
        self.attention = Attention(args['hidden_dim'], use_tanh=args['use_tanh'], C=args['tanh_exploration']) # Assuming Attention is defined elsewhere
        self.linear1 = nn.Linear(args['hidden_dim'], args['hidden_dim'])
        self.linear2 = nn.Linear(args['hidden_dim'], 1)

    def forward(self, hidden_state, encoder_outputs):
        e, logit = self.attention(hidden_state, encoder_outputs, use_tanh=self.args['use_tanh'], C=self.args['tanh_exploration'])  # Assuming attention returns context and attn_weights
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
    

# class MyNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(MyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)  # First dense layer
#         self.fc2 = nn.Linear(hidden_dim, 1)  # Second dense layer to reduce dimension to 1

#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # Apply ReLU activation function to the output of the first layer
#         x = self.fc2(x)  # Apply the second layer
#         return x.squeeze(1)  # Squeeze dimension 1 if it is of size 1



