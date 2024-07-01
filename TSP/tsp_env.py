import numpy as np
import torch
import torch.nn.functional as F
import sys

class State(torch.nn.Module):
    def __init__(self, mask):
        super(State, self).__init__()
        self.mask = mask
    
    def __repr__(self):
        return f'State(mask={self.mask})'


class VRPEnvironment:
    def __init__(self, args):
        """
        Initialize a new TSP environment.
        
        Inputs: 
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of tsp problems
                args['input_dim']: dimension of the problem which is 2 (coordinates)
        """
        self.args = args
        # How to change the batch_size as we go???
        self.input_data = torch.rand(self.args['batch_size'], self.args['n_nodes'], 2)

        self.input_pnt = self.input_data
        self.n_nodes = self.args['n_nodes']
        self.input_dim = self.args['input_dim']

    def reset(self, beam_width=1):
        """
        Reset the environment and return the initial state.
        """
        """
        # This definition is so bad... like the tensor wants to to be a flexible tensor 
        #       but pytorch doesnt work like that
        """
        self.beam_width = beam_width
        self.input_pnt = self.input_data
        batch_size = self.input_data.shape[0]

        self.mask = torch.zeros(batch_size, self.args['n_nodes'])  # self.mask: [batch_size, n_nodes]
        state = State(mask = self.mask)

        return state

    def step(self, idx, beam_parent=None):
        """
        Mask the nodes that can be visited in the next steps.
        """
        # Convert idx to a one-hot tensor and flatten
        # Remove the unnecessary dimension and ensure it's a long tensor
        # print('STEP idx shape:', idx.shape)
        idx = idx.squeeze(1).long()  # idx: [batch_size, 1] with indices of nodes
        
        # PyTorch one_hot requires the indices tensor to be in 'Long' datatype
        one_hot = F.one_hot(idx, num_classes=self.args['n_nodes'])

        # Add the one-hot tensor to the mask
        self.mask = self.mask + one_hot
        state = State(mask = self.mask)

        return state
    
def reward_func(route, show=False):
    """The reward for the TSP task is defined as the 
    negative value of the route length. This function gets the decoded
    actions and computed the reward.

    Args:
        route : tensor  shape [batch_size, n_nodes, input_dim] representing
            the coordinates of each point in the sequence of n_nodes for each batch.

    Returns:
        rewards (torch.Tensor): Reward tensor of size [batch_size] containing the 
            negative route length.
    """
    #Route shape: [batch_size, n_nodes, input_dim]
    # # Compute Euclidean distances between consecutive points, considering the route as circular
    # Calculate distances from each point to the next
    distances = torch.sqrt(torch.sum((route[:, 1:] - route[:, :-1]) ** 2, dim=2))
    # Sum distances for each route in the batch
    total_distance = torch.sum(distances, dim=1)
    # Compute distance from the last point back to the first to complete the loop
    closing_distance = torch.sqrt(torch.sum((route[:, 0] - route[:, -1]) ** 2, dim=1))
    # Add closing distance to total distance
    total_distance += closing_distance

    # Return negative route length as the reward
    return -total_distance
