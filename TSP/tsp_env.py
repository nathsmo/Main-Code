import numpy as np
import torch
import torch.nn.functional as F


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
                args['n_nodes']: number of packages in VRP
                args['input_dim']: dimension of the problem which is 2 (coordinates)
        """
        self.args = args
        # How to change the batch_size as we go???
        self.input_data = torch.rand(self.args['batch_size'], self.args['n_nodes'], 2)

        self.input_pnt = self.input_data
        self.n_nodes = self.args['n_nodes']
        self.input_dim = self.args['input_dim']

    def reset(self):
        """
        Reset the environment and return the initial state.
        """
        """
        # This definition is so bad... like the tensor wants to to be a flexible tensor 
        #       but pytorch doesnt work like that
        """
        self.input_pnt = self.input_data
        batch_size = self.input_data.shape[0]

        self.mask = torch.zeros(batch_size, self.args['n_nodes'])  # self.mask: [batch_size, n_nodes]
        state = State(mask = self.mask)

        return state

    def step(self, idx):
        """
        Mask the nodes that can be visited in the next steps.
        """

        # Convert idx to a one-hot tensor
        # Remove the unnecessary dimension and ensure it's a long tensor
        idx = idx.squeeze(1).long()  # idx: [batch_size, 1] with indices of nodes
        
        # PyTorch one_hot requires the indices tensor to be in 'Long' datatype
        one_hot = F.one_hot(idx, num_classes=self.args['n_nodes'])

        # Add the one-hot tensor to the mask
        self.mask = self.mask + one_hot
        state = State(mask = self.mask)

        return state
    
def reward_func(sample_solution=None):
    """The reward for the TSP task is defined as the 
    negative value of the route length. This function gets the decoded
    actions and computed the reward.

    Args:
        sample_solution : a list of tensors with len decode_len 
            each having a shape [batch_size x input_dim] representing
            the coordinates of each point in the sequence.

    Returns:
        rewards (torch.Tensor): Reward tensor of size [batch_size] containing the 
            negative route length.

    Example:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        decode_len = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    """
    # Stack the list of coordinate tensors to form a tensor of shape [decode_len, batch_size, input_dim]
    route = torch.stack(sample_solution)

    # Compute Euclidean distances between consecutive points, considering the route as circular
    distances = torch.norm(route - torch.roll(route, -1, 0), dim=2)

    # Calculate total distance for each route in the batch and negate to form the reward
    rewards = -distances.sum(0)

    return rewards

"""
# Example usage
sample_solution = [
    torch.tensor([[1.0, 1.0], [2.0, 2.0]]),  # Point sequence 1
    torch.tensor([[3.0, 3.0], [4.0, 4.0]]),  # Point sequence 2
    torch.tensor([[5.0, 5.0], [6.0, 6.0]])   # Point sequence 3
]
rewards = reward_func(sample_solution)
print("Rewards:", rewards)
"""