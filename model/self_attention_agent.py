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

from shared.embeddings import LinearEmbedding, EnhancedLinearEmbedding
from shared.self_decode_step import AttentionDecoder


# !!!!!! Decode type does not work in this code -> only greedy

class RLAgent(nn.Module):
    def __init__(self, args, env, dataGen, reward_func, clAttentionActor, clAttentionCritic, is_train=True):
        super(RLAgent, self).__init__()
        self.args = args
        self.env = env
        self.dataGen = dataGen
        self.reward_func = reward_func
        self.clAttentionCritic = clAttentionCritic
        self.clAttentionActor = clAttentionActor

        # Choose the type of embedding based on args
        self.embedding = LinearEmbedding(args['embedding_dim']) if args['emb_type'] == 'linear' else EnhancedLinearEmbedding(2, args['embedding_dim'])

        # Initialize the self-attention based decoder
        self.decodeStep = AttentionDecoder(input_dim=args['embedding_dim'], hidden_dim=args['hidden_dim'], 
                                           num_heads=args['num_heads'], num_actions=1)

        self.decoder_input = nn.Parameter(torch.randn(1, 1, args['embedding_dim']))
        init.xavier_uniform_(self.decoder_input)
        
        print("Agent created - Self Attention...")

    def build_model(self, decode_type): #prev -> forward
        args = self.args
        env = self.env
        input_pnt = env.input_pnt  # input_pnt: [batch_size x max_time x hidden_dim]
        batch_size = input_pnt.shape[0] 

        # encoder_emb_inp: [batch_size x seq_len x embedding_dim]
        context = self.embedding.forward(input_pnt)
        decoder_input = context[:, env.n_nodes - 1].unsqueeze(1) # decoder_input: [batch_size, 1, hidden_dim]
        
        # Reset the Environment.
        env.reset()


        for i in range(args['decode_len']):
            # Get logit and decoder_state

            # print("Dimensions of decoder input: [batch_size, seq, hidden_dim]", decoder_input.shape, 
            #         " ---context: ", context.shape, "mask: ", self.env.mask.shape)
            actions, logit, prob, attn_weights = self.decodeStep(decoder_input, context, self.env.mask)

            # logit, attn_weights = self.decodeStep.forward(decoder_input, context, self.env.mask)
            print("Logit shape: ", logit.shape)
            print("Prob shape: ", prob.shape)
            print("Actions: ", actions)

            sys.exit()

            # elif decode_type == "stochastic":
            #     # Select stochastic actions.
            #     prob_s = prob.squeeze(1)
            #     idx = torch.multinomial(prob_s, num_samples=1, replacement=True)
            # #     print("Idx: ", idx)

            # state = self.env.step(idx)
            # sys.exit()


            # Forward pass
            # output, attn_weights = model(inputs, context, mask)

            # print("Output shape:", output.shape)  # Should be [batch_size, hidden_dim]
            # print("Attention weights shape:", attn_weights.shape)  # Should be [batch_size, seq_length, seq_length]
            # print("Output:", output)

        # sys.exit()

        # print('*** THIS IS THE DECODER INPUT - encoder_emb_inp: ', encoder_emb_inp.shape)
        # sys.exit()
        # decoder_input = encoder_emb_inp[:, env.n_nodes - 1].unsqueeze(self.args['decode_len']) # decoder_input: [batch_size, seq_length, hidden_dim]
        # Decoding loop
        # sys.exit()
        # Decoder input: [batch_size x sequence_length x hidden_dim]
        # output, attn_weights = self.decodeStep(encoder_emb_inp, encoder_emb_inp, self.env)
        # output, attn_weights = self.decodeStep(output, output, self.env)

        print('------Output shape: ', output.shape, type(output))
        print("Output: ", output)
        probabilities = F.softmax(output, dim=-1)
        print("Probabilities: ", probabilities)

        sys.exit()
        for i in range(args['decode_len']):
            # Get logit and attention weights
            # print('decoder_input shape: ', decoder_input.shape)
            output, attn_weights = self.decodeStep(decoder_input, context, self.env)
            print('------Output shape: ', output.shape, type(output))
            probabilities = F.softmax(output, dim=-1)
    
            if decode_type == "greedy":
                action = torch.argmax(probabilities).item()  # Get the action with the maximum Q-value

            elif decode_type == "stochastic":  # Select stochastic actions.
                print("Probabilities: ", probabilities)
                action = torch.multinomial(probabilities, 1, replacement=True)
                print("Action: ", action)
                # idx = torch.multinomial(prob, num_samples=1, replacement=True)
            # print("Idx shape: ", idx.shape, type(idx))
            
            state = self.env.step(action)
            print("State: ", state)
            batched_idx = torch.cat([BatchSequence, action], dim=1).long()
            sys.exit()
            gathered = encoder_emb_inp[batched_idx[:, 0], batched_idx[:, 1]]
            # print('Gathered: ', gathered)
            # Expanding dimensions: Adding a dimension at axis 1
            decoder_input = gathered.unsqueeze(1)
            # print('decoder_input 2: ', decoder_input)
            # print("Decoder input shape: ", decoder_input.shape)
            # print("Decoder input mean: ", decoder_input.mean())
            
            # Advanced indexing in PyTorch to replace gather_nd
            selected_probs = probabilities[batched_idx[:, 0], batched_idx[:, 1]] 
            print("Selected probs: ", selected_probs)
            # Taking logarithm of the gathered elements
            log_prob = torch.log(selected_probs)

            probs.append(prob)
            idxs.append(idx)
            log_probs.append(log_prob)

            # Gather using the constructed indices
            action = input_pnt[batched_idx[:, 0], batched_idx[:, 1]]
            actions_tmp.append(action)
            actions = actions_tmp
            
            R = self.reward_func(actions)
            # print("Reward: ", R)
            
            # Critic
            v = torch.tensor(0)

            if decode_type == "stochastic":
                v = self.stochastic_process(batch_size, encoder_emb_inp)
        sys.exit()
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

        R, v, log_probs, actions, idxs , batch , probs = self.build_model("stochastic") 
        v_nograd = v.detach()
        R_nograd = R.detach()

        sys.exit()
        # Actor and Critic
        actor = self.clAttentionActor(self.args['hidden_dim'])
        critic = self.clAttentionCritic(self.args['hidden_dim'])
        # Losses
        actor_loss = torch.mean((R_nograd - v_nograd) * torch.sum(torch.stack(log_probs), dim=0))
        critic_loss = F.mse_loss(R_nograd, v)
        
        # Optimizers
        actor_optim = optim.Adam(actor.parameters(), lr=self.args['actor_net_lr'])
        critic_optim = optim.Adam(critic.parameters(), lr=self.args['critic_net_lr'])
        
        # Compute gradients
        # Clear previous gradients
        actor_optim.zero_grad()
        critic_optim.zero_grad()

        # Compute gradients
        actor_loss.backward(retain_graph=True) # Retain graph to compute gradients for critic
        critic_loss.backward(retain_graph=True)

        # # Clip gradients (optional, if args['max_grad_norm'] is set)
        # torch.nn.utils.clip_grad_norm_(actor.parameters(), self.args['max_grad_norm'])
        # torch.nn.utils.clip_grad_norm_(critic.parameters(), self.args['max_grad_norm'])

        # Apply gradients
        actor_optim.step()
        critic_optim.step()
        
        train_step = [actor_loss.item(), critic_loss.item(), R, v]
        
        return train_step

    def evaluate_single(self, eval_type="greedy"):

        start_time = time.time()
        avg_reward = []
        summary = self.build_model()

        self.dataGen.reset()
        test_df = self.dataGen.get_test_data()

        test_loader = DataLoader(test_df, batch_size=self.args['batch_size']) 

        problem_count = 0
        while problem_count < self.dataGen.n_problems:
            for data in test_loader:                
                self.env.input_data = data

                R, v, log_probs, actions, idxs, batch, _ = summary

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
                    
                    self.prt.print_out('\n\nVal-Step of {}: {}'.format(eval_type,step))
                    self.prt.print_out('\nExample test input: {}'.format(example_input))
                    self.prt.print_out('\nExample test output: {}'.format(example_output))
                    self.prt.print_out('\nExample test reward: {} - best: {}'.format(R[0],R_ind0))

                # Break the cycle
                problem_count += 1
                
                if problem_count >= self.dataGen.n_problems:
                    break
                
        end_time = time.time() - start_time
                
        # Finished going through the iterator dataset.
        self.prt.print_out('\n Validation overall avg_reward: {}'.format(np.mean(avg_reward)) )
        self.prt.print_out('Validation overall reward std: {}'.format(np.sqrt(np.var(avg_reward))) )

        self.prt.print_out("Finished evaluation with %d steps in %s." % (problem_count\
                           ,time.strftime("%H:%M:%S", time.gmtime(end_time))))
        
    
    def evaluate_batch(self, eval_type='greedy'):
        
        self.env.reset()

        summary = self.build_model()
        # summary = self.val_model_greedy(eval_type, "!!!!  evaluate batch !!! ")

        data = self.dataGen.get_test_data()
        
        start_time = time.time()

        if np.array_equal(self.env.input_data, data):
            print("The data is the same.!!!!!!")
            sys.exit()

        self.env.input_data = data

        R, v, log_probs, actions, idxs, batch, _ = summary

        if len(R.size()) == 0:
            print("This is the std of R: ", R.std())
            print("  R is empty !")
            sys.exit()

        std_r = R.std().item()
        # print("This is the std of R: ", std_r)
        # R = torch.min(R) 

        end_time = time.time() - start_time

        self.prt.print_out('Average of {} in batch-mode: {} -- std R: {} -- time {} s'.format(eval_type, R.mean().numpy(), str(std_r), end_time))  
        
    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
        
        elif infer_type == 'single':
            self.evaluate_single('greedy')
        
        self.prt.print_out("##################################################################")

    def run_train_step(self):
        data = self.dataGen.get_train_next()

        self.env.input_data = data
        
        train_results = self.build_train_step()

        return train_results
    


class MyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First dense layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Second dense layer to reduce dimension to 1

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function to the output of the first layer
        x = self.fc2(x)  # Apply the second layer
        return x.squeeze(1)  # Squeeze dimension 1 if it is of size 1
  