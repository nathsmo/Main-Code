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

import torch

def reward_func(sample_solution):
    """Compute the reward for a traveling salesman problem based on route length.
    
    Args:
        sample_solution (torch.Tensor): A tensor of shape [batch_size, n_nodes, input_dim]
            representing the coordinates of each point in the sequence for each batch.
    
    Returns:
        torch.Tensor: A tensor of size [batch_size] containing the negative route length.
    """
    # Calculate the Euclidean distance from each node to the next
    distances = torch.sqrt(torch.sum((sample_solution[:, 1:] - sample_solution[:, :-1])**2, dim=2))
    total_distance = torch.sum(distances, dim=1)
    
    # Calculate the distance from the last node back to the first to complete the loop
    closing_distance = torch.sqrt(torch.sum((sample_solution[:, 0] - sample_solution[:, -1])**2, dim=1))
    total_distance += closing_distance
    
    # Return negative distance as reward
    return -total_distance

# Example usage:
batch_size = 5
n_nodes = 3
input_dim = 2

# Mock data for sample_solution
# Shape [batch_size, n_nodes, input_dim]
# Example for two sample solutions in a batch, each with three nodes (coordinates)
sample_solution = torch.tensor([
    [[1.0, 1.0], [3.0, 3.0], [5, 5]],
    [[2.0, 2.0], [4.0, 4.0], [6, 6]],
    [[1.0, 1.0], [3.0, 3.0], [5, 5]],
    [[2.0, 2.0], [4.0, 4.0], [6, 6]],
    [[1.0, 1.0], [3.0, 3.0], [5, 5]]
])
print(sample_solution.size())
rewards = reward_func(sample_solution)
print('reward size:', rewards.size())
print("Rewards:", rewards)

# [
#     [[1,1],[2,2]],
# [[3,3],[4,4]],
# [[5,5],[6,6]]
# ]
