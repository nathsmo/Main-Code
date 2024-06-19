import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """ Implements a self-attention mechanism using nn.MultiheadAttention. """
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, query, key, value, mask=None):
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        attn_output, attn_weights = self.attention(query, key, value, key_padding_mask=mask)
        return attn_output.transpose(0, 1), attn_weights  # Convert back to (batch, seq_len, dim)

class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_actions, args, num_layers=1, dropout=0.1, beam_width=1):
        super(AttentionDecoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.self_attention = SelfAttention(hidden_dim, num_heads)
        self.action_head = nn.Linear(hidden_dim, num_actions)  # Action head for generating action logits
        self.rnn_layers = num_layers
        self.hidden_dim = hidden_dim
        self.beam_width = beam_width  # Beam width
        self.args = args

    def forward(self, inputs, context, mask=None):
        rnn_out, _ = self.rnn(inputs)
        attn_output, attn_weights = self.self_attention(rnn_out, context, context, mask)
        action_logits = self.action_head(attn_output[:, -1, :])  # Using the last timestep's output
        return action_logits, attn_weights

    def select_action(self, action_logits, method='greedy'):
        if method == 'beam_search':
            return self.beam_search(action_logits, self.beam_width)
        prob = F.softmax(action_logits, dim=-1)
        if method == "greedy":
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

        elif method == "stochastic":
            # Select stochastic actions.
            # print("Prob: ", prob.shape)
            idx = torch.multinomial(prob, num_samples=1, replacement=True)

        #Action selection
        return prob, idx 
        


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
