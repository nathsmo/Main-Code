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
        self.args = args

    def forward(self, inputs, context, mask=None):
        rnn_out, _ = self.rnn(inputs)
        attn_output, attn_weights = self.self_attention(rnn_out, context, context, mask)
        action_logits = self.action_head(attn_output[:, -1, :])  # Using the last timestep's output
        return action_logits, attn_weights
