import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class SelfAttention(nn.Module):
    """ Implements a self-attention mechanism using nn.MultiheadAttention. """
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=args['hidden_dim'], num_heads=args['num_heads'])

    def forward(self, query, key, value, mask=None):
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        attn_output, attn_weights = self.attention(query, key, value, key_padding_mask=mask)
        return attn_output.transpose(0, 1), attn_weights  # Convert back to (batch, seq_len, dim)

class AttentionDecoder(nn.Module):
    def __init__(self, attention_instance, num_actions, args):
        super(AttentionDecoder, self).__init__()
        self.rnn = nn.LSTM(input_size=args['embedding_dim'], hidden_size=args['hidden_dim'], num_layers=args['rnn_layers'],
                           batch_first=True, dropout=args['dropout'] if args['rnn_layers'] > 1 else 0)
        self.self_attention = attention_instance
        self.action_head = nn.Linear(args['hidden_dim'], num_actions)  # Action head for generating action logits
        self.beam_width = 1  # Beam width
        self.args = args

    def forward(self, inputs, context, mask=None):
        rnn_out, _ = self.rnn(inputs)
        attn_output, attn_weights = self.self_attention(rnn_out, context, context, mask)
        action_logits = self.action_head(attn_output[:, -1, :])  # Using the last timestep's output
        return action_logits, attn_weights

    def select_action(self, action_logits, visited_positions, method='greedy'):
        # This has a mask for previous visited nodes

        prob = F.softmax(action_logits, dim=-1)

        # Create a mask to set the probability of previously visited positions to zero
        mask = torch.ones_like(prob)
        mask.scatter_(1, visited_positions, 0)

        # Apply the mask to the probabilities
        masked_prob = prob * mask

        # Normalize the masked_prob to ensure it's a valid probability distribution
        masked_prob_sum = masked_prob.sum(dim=1, keepdim=True)
        
        # Check for invalid values and handle them
        if (masked_prob_sum == 0).any():
            print("Masked probabilities sum to zero, which means all actions are masked out. This should not happen.")
            print("Action logits:", action_logits)
            print("Probabilities:", prob)
            print("Masked probabilities:", masked_prob)
            print("First array Visited positions:", visited_positions[0])
            print("Last array Visited positions:", visited_positions[-1])
            raise RuntimeError("Masked probabilities are invalid.")
            sys.exit()
    
        masked_prob = masked_prob / masked_prob.sum(dim=1, keepdim=True)

        if torch.isnan(masked_prob).any() or torch.isinf(masked_prob).any():
            print("masked_prob contains NaN or Inf values")
            print("masked_prob:", masked_prob)
            raise ValueError("masked_prob contains NaN or Inf values")
        
        if (masked_prob < 0).any():
            print("masked_prob contains negative values")
            print("masked_prob:", masked_prob)
            raise ValueError("masked_prob contains negative values")

        if method == "greedy":
            # Epsilon-greedy action selection
            # Generate random numbers for each element in the batch
            random_values = torch.rand(masked_prob.size(0), 1, device=masked_prob.device)

            # Calculate the index of the maximum probability (greedy action)
            greedy_idx = torch.argmax(masked_prob, dim=1, keepdim=True)

            # Decide between the greedy action and a random action based on epsilon
            idx = torch.where(
                random_values < self.args['epsilon'],
                torch.randint(masked_prob.size(1), (masked_prob.size(0), 1), device=masked_prob.device),  # Random action
                greedy_idx  # Greedy action
            )
            self.args['epsilon'] *= 0.9999  # Decay epsilon
            # sys.exit()

        elif method == "stochastic":
            # Select stochastic actions.
            if masked_prob
            idx = torch.multinomial(masked_prob, num_samples=1, replacement=True)
        #Action selection
        return masked_prob, idx 
        


    def beam_search(self, logits, beam_width):
        # Start with an empty beam
        beam = [(torch.tensor([]), 0)]  # (action sequence, log_prob)
        for _ in range(logits.size(1)):  # Assuming logits are (batch_size, seq_len, num_actions)
            new_beam = []
            for prefix, score in beam:
                probabilities = F.softmax(logits, dim=-1)
                top_probs, top_inds = probabilities.topk(beam_width, dim=1)  # Get top beam_width probabilities and their indices
                for i in range(beam_width):
                    new_prefix = torch.cat([prefix, top_inds[:, i]])  # Append new index to prefix
                    new_score = score + torch.log(top_probs[:, i])  # Add log probability of choosing this index
                    new_beam.append((new_prefix, new_score))
            # Sort all candidates in new_beam by score in descending order and select the best beam_width ones
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        return beam[0][0]  # Return the sequence with the highest probability

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.rnn_layers, batch_size, self.hidden_dim),
                torch.zeros(self.rnn_layers, batch_size, self.hidden_dim))
