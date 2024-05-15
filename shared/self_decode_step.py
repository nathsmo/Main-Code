
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
    def __init__(self, input_dim, hidden_dim, num_heads, num_actions, num_layers=1, dropout=0.1):
        super(AttentionDecoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.self_attention = SelfAttention(hidden_dim, num_heads)
        self.action_head = nn.Linear(hidden_dim, num_actions)  # Action head for generating action logits
        self.rnn_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, inputs, context, mask=None):
        rnn_out, _ = self.rnn(inputs)
        attn_output, attn_weights = self.self_attention(rnn_out, context, context, mask)
        action_logits = self.action_head(attn_output[:, -1, :])  # Using the last timestep's output
        return action_logits, attn_weights

    def select_action(self, action_logits, method='greedy'):
        probabilities = F.softmax(action_logits, dim=-1)
        if method == 'greedy':
            return probabilities, torch.argmax(probabilities, dim=-1)  # Greedy selection
        elif method == 'stochastic':
            return probabilities, torch.multinomial(probabilities, 1).squeeze(-1)  # Stochastic selection

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.rnn_layers, batch_size, self.hidden_dim),
                torch.zeros(self.rnn_layers, batch_size, self.hidden_dim))
    
# Example usage
batch_size = 128
seq_length = 10
hidden_dim = 128
num_heads = 4

# # Instantiate the decoder
# attention_decoder = AttentionDecoder(input_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads)

# # Forward pass
# logit, attn_weights = attention_decoder(inputs, context, mask)
# print("Logits:", logit)
# print("Logits shape:", logit.shape)
# print("Attention weights shape:", attn_weights.shape)



# # Example usage
# num_actions = 5  # Define the number of possible actions
# attention_decoder = AttentionDecoder(input_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_actions=num_actions)

# # Dummy data
# inputs = torch.randn(batch_size, seq_length, hidden_dim)
# context = torch.randn(batch_size, seq_length, hidden_dim)

# # Forward pass
# action_logits, attn_weights = attention_decoder(inputs, context)
# # print("Action logits:", action_logits)
# print("Action logits shape:", action_logits.shape)

# # Action selection
# selected_action_greedy = attention_decoder.select_action(action_logits, 'greedy')
# selected_action_stochastic = attention_decoder.select_action(action_logits, 'stochastic')
# print("Selected action (greedy) shape:", selected_action_greedy.shape)
# print("Selected action (stochastic) shape:", selected_action_stochastic.shape)

# print("Selected action (greedy):", selected_action_greedy)
# print("Selected action (stochastic):", selected_action_stochastic)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SelfAttention(nn.Module):
#     """ Implements a self-attention mechanism using nn.MultiheadAttention. """
#     def __init__(self, hidden_dim, num_heads):
#         super(SelfAttention, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

#     def forward(self, query, key, value, mask=None):
#         # Ensure input is in the correct format (seq_len, batch, dim)
#         query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
#         attn_output, attn_weights = self.attention(query, key, value, key_padding_mask=mask)
#         attn_output = attn_output.transpose(0, 1)  # Convert back to (batch, seq_len, dim)
#         return attn_output, attn_weights

# class AttentionDecoder(nn.Module):
#     """ Decoder with RNN, self-attention, and action selection. """
#     def __init__(self, input_dim, hidden_dim, num_heads, num_layers=1, dropout=0.1):
#         super(AttentionDecoder, self).__init__()
#         self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
#                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
#         self.self_attention = SelfAttention(hidden_dim, num_heads)

#     def forward(self, inputs, context, mask=None):
#         # Initial RNN processing
#         rnn_out, _ = self.rnn(inputs)
#         # Self-attention processing
#         attn_output, attn_weights = self.self_attention(rnn_out, context, context, mask)
#         return attn_output, attn_weights

# # Example usage
# batch_size = 128
# seq_length = 10
# hidden_dim = 128
# num_heads = 4

# # Instantiate the decoder
# attention_decoder = AttentionDecoder(input_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads)

# # Dummy data
# inputs = torch.randn(batch_size, seq_length, hidden_dim)
# context = torch.randn(batch_size, seq_length, hidden_dim)
# mask = None  # Define as needed

# # Forward pass
# logit, attn_weights = attention_decoder(inputs, context, mask)
# print("Logits:", logit)
# print("Logits shape:", logit.shape)
# print("Attention weights shape:", attn_weights.shape)


# # # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # class SelfAttention(nn.Module):
# # #     """ Implements a self-attention mechanism using nn.MultiheadAttention. """
# # #     def __init__(self, hidden_dim, num_heads):
# # #         super(SelfAttention, self).__init__()
# # #         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

# # #     def forward(self, query, key, value, mask=None):
# # #         # Ensure input is in the correct format (seq_len, batch, dim)
# # #         query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
# # #         attn_output, attn_weights = self.attention(query, key, value, key_padding_mask=mask)
# # #         attn_output = attn_output.transpose(0, 1)  # Convert back to (batch, seq_len, dim)
        
# # #         return attn_output, attn_weights

# # # class AttentionDecoder(nn.Module):
# # #     """ Decoder with RNN and self-attention. """
# # #     def __init__(self, input_dim, hidden_dim, num_heads, num_layers=1, dropout=0.1):
# # #         super(AttentionDecoder, self).__init__()
# # #         self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
# # #                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
# # #         self.self_attention = SelfAttention(hidden_dim, num_heads)
# # #         self.hidden_dim = hidden_dim

# # #     def forward(self, inputs, context, mask=None):
# # #         # Initial RNN processing
# # #         rnn_out, hidden = self.rnn(inputs)
# # #         rnn_out = rnn_out[:, -1, :].unsqueeze(1)  # Get the last time step output

# # #         # Self-attention processing
# # #         attn_output, attn_weights = self.self_attention(rnn_out, context, context, mask)
# # #         return attn_output, attn_weights

# # #     # def _init_hidden(self, batch_size):
# # #     #     return (torch.zeros(self.rnn_layers, batch_size, self.hidden_dim),
# # #     #             torch.zeros(self.rnn_layers, batch_size, self.hidden_dim))

# # # # Example usage
# # # batch_size = 64
# # # seq_length = 10
# # # hidden_dim = 64
# # # num_heads = 4

# # # # Create model instance
# # # model = AttentionDecoder(input_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads)

# # # # Create sample input tensors
# # # inputs = torch.randn(batch_size, seq_length, hidden_dim)  # [batch_size, seq_length, hidden_dim]
# # # context = torch.randn(batch_size, seq_length, hidden_dim)  # Same shape as inputs
# # # mask = torch.zeros(batch_size, seq_length).bool()  # Example mask, assuming no padding

# # # # Forward pass
# # # output, attn_weights = model(inputs, context, mask)

# # # print("Output shape:", output.shape)  # Should be [batch_size, hidden_dim]
# # # print("Attention weights shape:", attn_weights.shape)  # Should be [batch_size, seq_length, seq_length]
# # # # print("Output:", output)