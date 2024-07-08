import torch
import torch.nn as nn
import torch.optim as optim

# # Define a simple neural network
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(10, 5)  # 10 input features, 5 output features

#     def forward(self, x):
#         return self.fc1(x)

# # Create an instance of the network
# net = SimpleNet()

# # Define an optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)

# # Define a simple loss function
# loss_function = nn.MSELoss()

# # Sample data: input and target output
# input_data = torch.randn(10, 10)  # batch of 10, 10 features each
# target_data = torch.randn(10, 5)  # batch of 10, 5 target features each

# # Training loop
# for epoch in range(5):  # loop over the dataset multiple times
#     optimizer.zero_grad()   # zero the parameter gradients
#     outputs = net(input_data)  # forward pass
#     loss = loss_function(outputs, target_data)  # calculate loss
#     loss.backward()  # backward pass
#     optimizer.step()  # update parameters
    
#     print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# # After this loop, `net` retains updated weights and can be used for further processing or evaluation

# import torch

# def reward_func(sample_solution):
#     """Compute the reward for a traveling salesman problem based on route length.
    
#     Args:
#         sample_solution (torch.Tensor): A tensor of shape [batch_size, n_nodes, input_dim]
#             representing the coordinates of each point in the sequence for each batch.
    
#     Returns:
#         torch.Tensor: A tensor of size [batch_size] containing the negative route length.
#     """
#     # Calculate the Euclidean distance from each node to the next
#     distances = torch.sqrt(torch.sum((sample_solution[:, 1:] - sample_solution[:, :-1])**2, dim=2))
#     total_distance = torch.sum(distances, dim=1)
    
#     # Calculate the distance from the last node back to the first to complete the loop
#     closing_distance = torch.sqrt(torch.sum((sample_solution[:, 0] - sample_solution[:, -1])**2, dim=1))
#     total_distance += closing_distance
    
#     # Return negative distance as reward
#     return -total_distance

# # Example usage:
# batch_size = 5
# n_nodes = 3
# input_dim = 2

# # Mock data for sample_solution
# # Shape [batch_size, n_nodes, input_dim]
# # Example for two sample solutions in a batch, each with three nodes (coordinates)
# sample_solution = torch.tensor([
#     [[1.0, 1.0], [3.0, 3.0], [5, 5]],
#     [[2.0, 2.0], [4.0, 4.0], [6, 6]],
#     [[1.0, 1.0], [3.0, 3.0], [5, 5]],
#     [[2.0, 2.0], [4.0, 4.0], [6, 6]],
#     [[1.0, 1.0], [3.0, 3.0], [5, 5]]
# ])
# print(sample_solution.size())
# rewards = reward_func(sample_solution)
# print('reward size:', rewards.size())
# print("Rewards:", rewards)

# # [
# #     [[1,1],[2,2]],
# # [[3,3],[4,4]],
# # [[5,5],[6,6]]
# # ]


# import torch

# def reward_func(route):
#     """
#     The reward for the TSP task is defined as the negative value of the route length.
#     This function gets the decoded actions and computes the reward.

#     Args:
#         route: tensor of shape [batch_size, n_nodes, input_dim]

#     Returns:
#         rewards: tensor of size [batch_size]
#     """
    
#     # Tilt the sample solution
#     route_tilted = torch.cat((route[:, -1:, :], route[:, :-1, :]), dim=1)
#     print(route_tilted)

#     # Calculate route lengths
#     route_lens_decoded = torch.sum(torch.sqrt(torch.sum(
#         (route_tilted - route) ** 2, dim=2)), dim=1)
    
#     return route_lens_decoded

# # Example usage:
# route = torch.tensor([[[1, 1], [3, 3]], [[2, 2], [4, 4]]])
# rewards = reward_func(route)
# print(rewards)



# import torch

# def reward_func(route):
#     """
#     The reward for the TSP task is defined as the negative value of the route length.
#     This function gets the decoded actions and computes the reward.

#     Args:
#         route: tensor of shape [batch_size, n_nodes, input_dim]

#     Returns:
#         rewards: tensor of size [batch_size]
#     """
    
#     # Tilt the sample solution by moving the last node to the first position
#     route_tilted = torch.cat((route[:, -1:, :], route[:, :-1, :]), dim=1)

#     # Calculate Euclidean distances between consecutive points
#     distances = torch.sqrt(torch.sum((route_tilted - route) ** 2, dim=2))

#     # Sum the distances along the nodes to get the total route length for each batch
#     route_lens_decoded = torch.sum(distances, dim=1)
    
#     return -route_lens_decoded

# # Example usage:
# route = torch.rand(128, 10, 2)  # Random example tensor with shape [128, 10, 2]
# rewards = reward_func(route)
# print(rewards)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Attention(nn.Module):
#     def __init__(self, hidden_dim, use_tanh=False, C=10):
#         super(Attention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.use_tanh = use_tanh
#         self.C = C

#         self.W_query = nn.Linear(hidden_dim, hidden_dim)
#         self.W_ref = nn.Linear(hidden_dim, hidden_dim)
#         self.v = nn.Linear(hidden_dim, 1)

#     def forward(self, query, ref, mask=None):
#         # query: [batch_size, hidden_dim]
#         # ref: [batch_size, seq_len, hidden_dim]
        
#         # Expand query to [batch_size, seq_len, hidden_dim]
#         query_expanded = query.unsqueeze(1).expand(-1, ref.size(1), -1)
        
#         # Compute scores
#         scores = self.v(torch.tanh(self.W_query(query_expanded) + self.W_ref(ref)))
#         scores = scores.squeeze(-1)  # [batch_size, seq_len]

#         if self.use_tanh:
#             scores = self.C * torch.tanh(scores)
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
        
#         attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        
#         # Compute context vector
#         context = torch.bmm(attn_weights.unsqueeze(1), ref).squeeze(1)  # [batch_size, hidden_dim]
        
#         return context, attn_weights


# class Critic(nn.Module):
#     def __init__(self, args):
#         super(Critic, self).__init__()
#         self.args = args
#         self.attention = Attention(args['hidden_dim'], use_tanh=args['use_tanh'], C=args['tanh_exploration'])
#         self.linear1 = nn.Linear(args['hidden_dim'], args['hidden_dim'])
#         self.linear2 = nn.Linear(args['hidden_dim'], 1)

#     def forward(self, hidden_state, encoder_outputs):
#         e, logit = self.attention(hidden_state, encoder_outputs)
#         return e
    
#     def final_step_critic(self, hy):
#         x = torch.relu(self.linear1(hy))
#         x = self.linear2(x)
#         v = x.squeeze(1)
#         return v

# def reward_func(sample_solution):
#     sample_solution_tilted = torch.cat((sample_solution[:, -1:, :], sample_solution[:, :-1, :]), dim=1)
#     distances = torch.sqrt(torch.sum((sample_solution_tilted - sample_solution) ** 2, dim=2))
#     route_lens_decoded = torch.sum(distances, dim=1)
#     return -route_lens_decoded

# # Example usage
# args = {
#     'hidden_dim': 128,
#     'rnn_layers': 2,
#     'use_tanh': True,
#     'tanh_exploration': 10,
#     'n_process_blocks': 3
# }
# batch_size = 128
# input_dim = 2
# sample_solution = torch.rand(batch_size, 10, input_dim)
# rewards = reward_func(sample_solution)

# # Example critic usage
# critic = Critic(args)
# hidden_state = torch.zeros(batch_size, args['hidden_dim'])
# cell_state = torch.zeros(args['rnn_layers'], batch_size, args['hidden_dim'])

# # Assuming action_selected should have hidden_dim as its last dimension
# action_selected = torch.randn(batch_size, args['hidden_dim'])  # Random tensor with shape [batch_size, hidden_dim]
# action4critic = action_selected.unsqueeze(0)  # Add an extra dimension to match LSTM input
# print("action4critic:", action4critic.size())
# print("hidden_state:", hidden_state.size())
# print("cell_state:", cell_state.size())
# print("rnn_layers:", args['rnn_layers'])
# lstm_layer = nn.LSTM(input_size=args['hidden_dim'], hidden_size=args['hidden_dim'], num_layers=args['rnn_layers'])
# output, (hn, cn) = lstm_layer(action4critic, (hidden_state.unsqueeze(0), cell_state))

# hy = hn[-1]
# context = sample_solution  # Placeholder for context
# idxs = torch.randint(0, 10, (batch_size, 1))  # Placeholder for idxs

# for i in range(args['n_process_blocks']):
#     hy = critic(hy, context)

# v = critic.final_step_critic(hy)
# print("v:", v)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sys

# class Critic(nn.Module):
#     def __init__(self, args):
#         super(Critic, self).__init__()
#         self.args = args
#         self.attention_layers = nn.ModuleList([Attention(args['hidden_dim']) for _ in range(args['n_process_blocks'])])
#         self.linear1 = nn.Linear(args['hidden_dim'], args['hidden_dim'])
#         self.linear2 = nn.Linear(args['hidden_dim'], 1)

#     def forward(self, encoder_outputs, batch_size):
#         # Initialize LSTM states
#         c_t = [torch.zeros(batch_size, self.args['hidden_dim']) for _ in range(self.args['rnn_layers'])]
#         hy = c_t[0]
#         # Process blocks
#         for i, attention_layer in enumerate(self.attention_layers):
#             # Attention layer input = hidden state, encoded input
#             #   hy should be: [batch_size, input_dim]
#             #   encoder_outputs should be: [batch_size, seq_len, input_dim]
#             print('hy:', hy.size(), 'encoder_outputs:', encoder_outputs.size())
#             action_Logits, attentionW = attention_layer(hy, encoder_outputs)
#             print('attentionW:', attentionW.size(), 'action_Logits:', action_Logits.size())
#             prob_c = F.softmax(attentionW, dim=-1)
#             print('prob_c:', prob_c)
#             sys.exit()
#             print("e:", e_c.size())
#             e_c = e_c.expand(e_c.size(0), -1, -1)
#             hy = torch.bmm(prob_c, e_c)#.squeeze(1)
#             sys.exit()

#         # Linear layers
#         x = F.relu(self.linear1(hy))
#         v = self.linear2(x).squeeze(1)
#         return v

# class Attention(nn.Module):
#     """A generic attention module for a decoder in seq2seq models"""
#     def __init__(self, dim, use_tanh=False, C=10):
#         super(Attention, self).__init__()
#         self.use_tanh = use_tanh        # Set use_tanh flag
#         self.C = C          # Set scaling factor C # tanh exploration parameter
#         self.tanh = nn.Tanh() # Define hyperbolic tangent function

#         # Define linear layers for projection
#         self.project_query = nn.Linear(dim, dim)
#         self.project_ref = nn.Linear(dim, dim)
#         #Could've been this too:         self.project_ref = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
#         self.v = nn.Parameter(torch.randn(1, dim, 1))


#     def forward(self, query, ref, *args, **kwargs): # called __call__ before
#         # Project query tensor
#         q = self.project_query(query)  # [batch_size x dim]
#         # Apply convolution along the time dimension
#         e = self.project_ref(ref)  # [batch_size x max_time x dim]
#         if e.shape[1] > e.shape[0]:
#             e = e.permute(1, 0, 2)
#         # Expand dimensions of q to match the shape of e
#         if q.dim() == 2:
#             q = q.unsqueeze(1) # [batch_size, max_time, dim]
#         # Prepare v for batch multiplication
#         v_view = self.v.expand(e.size(0), -1, -1)  # [batch_size x dim x 1]
#         # Apply tanh activation function over the sum, prepare for multiplication
#         tanh_output = self.tanh(q + e)
#         if tanh_output.dim() > 3:
#             print("There's something wrong inside the critic network - attention file")
#         # Compute the attention logits using batch matrix multiplication
#         u = torch.bmm(tanh_output, v_view).squeeze(2)  # [batch_size x max_time]
#         # Apply scaling factor and tanh if required
#         if self.use_tanh:
#             logits = self.C * self.tanh(u)
#         else:
#             logits = u

#         return e, logits

# # Example usage
# args = {
#     'hidden_dim': 128,
#     'rnn_layers': 2,
#     'n_process_blocks': 3
# }
# batch_size = 128
# seq_len = 10
# input_dim = 2
# encoder_outputs = torch.rand(batch_size, seq_len, args['hidden_dim'])

# critic = Critic(args)
# v = critic(encoder_outputs, batch_size)
# print("v:", v)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.C = C

        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_ref = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query, ref, mask=None):
        # query: [batch_size, hidden_dim]
        # ref: [batch_size, seq_len, hidden_dim]

        # Expand query to [batch_size, seq_len, hidden_dim]
        query_expanded = query.unsqueeze(1).expand(-1, ref.size(1), -1)

        # Compute scores
        scores = self.v(torch.tanh(self.W_query(query_expanded) + self.W_ref(ref)))
        scores = scores.squeeze(-1)  # [batch_size, seq_len]

        if self.use_tanh:
            scores = self.C * torch.tanh(scores)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]

        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), ref).squeeze(1)  # [batch_size, hidden_dim]

        return context, attn_weights

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.attention = Attention(args['hidden_dim'], use_tanh=args['use_tanh'], C=args['tanh_exploration'])
        self.fc1 = nn.Linear(args['hidden_dim'], args['hidden_dim'])
        self.fc2 = nn.Linear(args['hidden_dim'], 1)

    def forward(self, hidden_state, encoder_outputs):
        # hidden_state: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]

        # Apply attention mechanism
        context, attn_weights = self.attention(hidden_state, encoder_outputs)

        # Process context vector to compute value estimate
        x = F.relu(self.fc1(context))
        v = self.fc2(x).squeeze(-1)  # [batch_size]

        return v

# Example usage
args = {
    'hidden_dim': 128,
    'rnn_layers': 2,
    'use_tanh': True,
    'tanh_exploration': 10,
    'n_process_blocks': 3
}
batch_size = 128
seq_len = 10
input_dim = 2
encoder_outputs = torch.rand(batch_size, seq_len, args['hidden_dim'])
# print("encoder_outputs:", encoder_outputs)

critic = Critic(args)
hidden_state = torch.zeros(batch_size, args['hidden_dim'])
v = critic(hidden_state, encoder_outputs)
print("v:", v)



