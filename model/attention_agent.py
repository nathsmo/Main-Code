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

from shared.embeddings import LinearEmbedding, EnhancedLinearEmbedding, Enhanced__LinearEmbedding, MinimalLinearEmbedding
from shared.decode_step import RNNDecodeStep

class RLAgent(nn.Module):

    def __init__(self, args, prt, env, dataGen, reward_func, clAttentionActor, clAttentionCritic, is_train=True):
        super().__init__()

        self.args = args
        self.prt = prt
        self.env = env
        self.dataGen = dataGen
        self.reward_func = reward_func
        self.clAttentionCritic = clAttentionCritic
        self.clAttentionActor = clAttentionActor

        # Embedding and Decoder setup
        if args['emb_type'] == 'linear':
            self.embedding = LinearEmbedding(prt, args['embedding_dim'])
        elif args['emb_type'] == 'enhanced':
            self.embedding = EnhancedLinearEmbedding(prt, 2, args['embedding_dim'])
        elif args['emb_type'] == 'enhanced2':
            self.embedding = Enhanced__LinearEmbedding(prt, 2, args['embedding_dim'])
        elif args['emb_type'] == 'minimal':
            self.embedding = MinimalLinearEmbedding(prt, 2, args['embedding_dim'])
            
        self.decodeStep = RNNDecodeStep(clAttentionActor, args['hidden_dim'],
                                        use_tanh=args['use_tanh'],
                                        tanh_exploration=args['tanh_exploration'],
                                        n_glimpses=args['n_glimpses'],
                                        mask_glimpses=args['mask_glimpses'],
                                        mask_pointer=args['mask_pointer'],
                                        forget_bias=args['forget_bias'],
                                        rnn_layers=args['rnn_layers'])
        
        self.decoder_input = nn.Parameter(torch.randn(1, 1, args['embedding_dim']))
        init.xavier_uniform_(self.decoder_input)
        self.prt.print_out("Agent created - Pointer Network.")

       # Initialize actor and critic networks
        self.actor = clAttentionActor(self.args['hidden_dim'])
        self.critic = clAttentionCritic(self.args['hidden_dim'])

        # Define optimizers for actor and critic
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.args['actor_net_lr'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.args['critic_net_lr'])


    def build_model(self, decode_type= "greedy"):
        # Builds the model
        args = self.args
        env = self.env
        batch_size = env.input_pnt.shape[0]
        
        if decode_type == 'beam_search':
            beam_width = args['beam_width']
        else:
            beam_width = 1

        # print("Batch size: ", batch_size)
        # print("env.input_pnt shape: ", env.input_pnt.shape)

        # input_pnt: [batch_size x max_time x input_dim=2]
        input_pnt = env.input_pnt
        
        if not isinstance(input_pnt, torch.Tensor):
            input_pnt = torch.tensor(input_pnt, dtype=torch.float).float()
        # encoder_emb_inp: [batch_size x max_time x embedding_dim]
        encoder_emb_inp = self.embedding.forward(input_pnt)

        # Reset the Environment.
        env.reset(beam_width)

        # Create tensors and lists
        actions_tmp = []
        log_probs = []
        probs = []
        idxs = []

        if beam_width > 1:
            BatchSequence = torch.arange(0, batch_size * beam_width, dtype=torch.int64).unsqueeze(1)
            # Start from depot
            idx = (env.n_nodes - 1) * torch.ones(batch_size * beam_width, 1, dtype=torch.int64)

            # Assuming 'input_pnt' is a PyTorch tensor and is correctly dimensioned
            # Repeat the last node's input points to match beam width
            # Ensure that input_pnt[:, env.n_nodes - 1] is 2D by unsqueezing if necessary
            if input_pnt[:, env.n_nodes - 1].dim() == 1:
                input_pnt = input_pnt[:, env.n_nodes - 1].unsqueeze(1)  # Make it [batch_size, 1] if it's not
            action = input_pnt.repeat(1, beam_width, 1)
            decoder_state = self.decodeStep._init_hidden(batch_size*beam_width)
            decoder_input = self.decoder_input.repeat(batch_size * beam_width, 1, 1)
            context = encoder_emb_inp.repeat(beam_width, 1, 1)

        else:
            BatchSequence = torch.arange(batch_size, dtype=torch.int64).unsqueeze(1)
            # Start from depot
            idx = (env.n_nodes - 1) * torch.ones([batch_size, 1])
            action = input_pnt[:, env.n_nodes - 1].unsqueeze(1)

            # Decoder State
            # Initialize LSTM state tensors for hidden (h) and cell (c) states
            decoder_state = self.decodeStep._init_hidden(batch_size)
            # print('---Shape of encoder_embedding input: ', encoder_emb_inp.shape)
            # Start from trainable nodes in TSP
            decoder_input = encoder_emb_inp[:, env.n_nodes - 1].unsqueeze(1) # decoder_input: [batch_size, 1, hidden_dim]

            # Decoding loop
            context = encoder_emb_inp

        
        for i in range(args['decode_len']):
            # Get logit and decoder_state
            logit, decoder_state = self.decodeStep.forward(decoder_input, context, self.env, decoder_state)
            # print('Logit shape: ', logit.shape)
            prob = F.softmax(logit, dim=-1)
            # print("Probabilities: ", prob)
            beam_parent = None

            if decode_type == "greedy":
                # Generate random numbers for each element in the batch
                random_values = torch.rand(prob.size(0))

                # Calculate the index of the maximum probability (greedy action)
                greedy_idx = torch.argmax(prob, dim=1).unsqueeze(1)

                random_values = torch.rand(prob.size(0), 1)
                # Decide between the greedy action and a random action
                idx = torch.where(random_values < self.args['epsilon'],
                  torch.randint(prob.size(1), (prob.size(0), 1), device=prob.device),  # Random action
                  greedy_idx)   # Greedy action
                self.args['epsilon'] *= 0.9999  # Decay epsilon

            elif decode_type == "stochastic":
                # Select stochastic actions.
                # print("Prob: ", prob.shape)
                idx = torch.multinomial(prob, num_samples=1, replacement=True)

            elif decode_type == "beam_search":
                if i == 0:
                    # Create a sequence for batch_beam index calculation
                    batchBeamSeq = torch.arange(batch_size).unsqueeze(1).repeat(1, beam_width).view(-1, 1)
                    beam_path = []
                    log_beam_probs = []
                    # Log probabilities for selecting beam_width different branches initially
                    log_beam_prob = torch.log(torch.chunk(prob, beam_width, dim=0)[0])

                elif i > 0:
                    log_beam_prob = torch.log(prob) + log_beam_probs[-1]
                    # Reshape to align for beam_width grouping
                    log_beam_prob = torch.cat(torch.chunk(log_beam_prob, beam_width, dim=0), dim=1)

                # Get top k probabilities and indices
                topk_logprob_val, topk_logprob_ind = torch.topk(log_beam_prob, beam_width, dim=1)

                # Reshape for correct beam search tracking
                topk_logprob_val = topk_logprob_val.transpose(0, 1).contiguous().view(-1, 1)
                topk_logprob_ind = topk_logprob_ind.transpose(0, 1).contiguous().view(-1, 1)

                # Compute new indices for the next step and trace back the beam parents
                idx = (topk_logprob_ind % env.n_nodes).long()  # Which city in the route.
                beam_parent = (topk_logprob_ind // env.n_nodes).long()  # Which hypothesis it came from.

                # Update batchBeamSeq for next usage and gather new probs
                batchedBeamIdx = batchBeamSeq + batch_size * beam_parent
                prob = prob.view(-1)[batchedBeamIdx.squeeze()]

                beam_path.append(beam_parent)
                log_beam_probs.append(topk_logprob_val)
                #------ ending of beam search

            state = self.env.step(idx, beam_parent)
            # print("State: ", state)
            batched_idx = torch.cat([BatchSequence, idx], dim=1).long()
            if decode_type == "beam_search":
                encoder_emb_inp = encoder_emb_inp.repeat(beam_width, 1, 1)

            gathered = encoder_emb_inp[batched_idx[:, 0], batched_idx[:, 1]]

            # Expanding dimensions: Adding a dimension at axis 1
            decoder_input = gathered.unsqueeze(1)

            # Advanced indexing in PyTorch to replace gather_nd
            selected_probs = prob[batched_idx[:, 0], batched_idx[:, 1]]  # Adjust this based on your actual index structure
            # print("Selected probs: ", selected_probs)
            # Taking logarithm of the gathered elements
            log_prob = torch.log(selected_probs)

            probs.append(prob)
            idxs.append(idx)
            log_probs.append(log_prob)

            if decode_type == "beam_search":
                input_pnt = input_pnt.repeat(beam_width, 1, 1)

            # Gather using the constructed indices
            action = input_pnt[batched_idx[:, 0], batched_idx[:, 1]]
            actions_tmp.append(action)
            if decode_type == "beam_search":
                tmplst = []
                tmpind = [BatchSequence]  # Ensure BatchSequence is properly initialized, e.g., torch.arange(batch_size * beam_width).view(-1, 1)
                for k in reversed(range(len(actions_tmp))):
                    # Gather from actions_tmp using the last element of tmpind as indices
                    current_indices = tmpind[-1].squeeze()  # Remove any singleton dimensions for indexing
                    selected_actions = actions_tmp[k][current_indices]
                    tmplst = [selected_actions] + tmplst

                    # Update indices for the next iteration
                    new_indices = batchBeamSeq[current_indices] + batch_size * beam_path[k][current_indices]
                    tmpind.append(new_indices)

                actions = tmplst
            else:
                actions = actions_tmp

            R = self.reward_func(actions)
            # print("Reward: ", R)

            # Critic
            v = torch.tensor(0)

            if decode_type == "stochastic":
                v = self.stochastic_process(batch_size, encoder_emb_inp)
        return (R, v, log_probs, actions, idxs, self.env.input_pnt , probs)
    
    def stochastic_process(self, batch_size, encoder_emb_inp):
        # Init States
        (h, c) = self.decodeStep._init_hidden(batch_size)
        hy = h[0]

        for i in range(self.args['n_process_blocks']):
            process = self.clAttentionCritic(self.args['hidden_dim'])
            e, logit = process(hy, encoder_emb_inp, self.env)

            prob = torch.softmax(logit, dim=-1)

            # hy: [batch_size x 1 x sourceL] * [batch_size x sourceL x hidden_dim] -> 
            #       [batch_size x hidden_dim]
            prob_expanded = prob.unsqueeze(1)

            # Perform matrix multiplication
            hy = torch.matmul(prob_expanded, e)

            # Final shape of 'hy' will be [batch_size, hidden_dim]
            hy = hy.squeeze(1)

            input_dim = hy.size(1)  # Assuming 'hy' is of shape [batch_size, feature_size]

            network = MyNetwork(input_dim, self.args['hidden_dim'])

            # Forward pass to compute 'v'
            v = network(hy)

        return v


    def build_train_step(self):
        """
        This function returns a train_step op, in which by running it we proceed one training step.
        """

        R, v, log_probs, actions, idxs , batch , probs = self.build_model("stochastic") # self.train_model

        R = R.float()
        v = v.float()
        
        v_nograd = v.detach()
        R_nograd = R.detach()

        # Losses
        actor_loss = torch.mean((R_nograd - v_nograd) * torch.sum(torch.stack(log_probs), dim=0))
        critic_loss = F.mse_loss(R_nograd, v)
    
        # Compute gradients
        # Clear previous gradients
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # Compute gradients
        actor_loss.backward(retain_graph=True) # Retain graph to compute gradients for critic
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
        data = self.dataGen.get_train_next()
        self.env.input_data = data
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
    

class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First dense layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Second dense layer to reduce dimension to 1

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function to the output of the first layer
        x = self.fc2(x)  # Apply the second layer
        return x.squeeze(1)  # Squeeze dimension 1 if it is of size 1