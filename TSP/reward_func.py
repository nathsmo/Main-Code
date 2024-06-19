import torch

def _compute_route_length(sample_solution):
    """
    Compute the length of the route based on the sample_solution.

    Args:
        sample_solution (torch.Tensor): A tensor of shape [batch_size, decode_len, input_dim]
            representing the coordinates of each point in the sequence for each sample in the batch.

    Returns:
        torch.Tensor: A tensor of size [batch_size] containing the route length for each sample.
    """
    # Calculate differences between consecutive points
    rolled_solution = torch.roll(sample_solution, shifts=-1, dims=1)
    distances = torch.norm(sample_solution - rolled_solution, dim=2)
    print('Distances reward func file:', distances)

    zero_count = distances.count(0)
    if zero_count > 0:
        print("Number of zeros in the list:", zero_count)
    # Sum the distances for each route in the batch to get the total length of each route
    route_lengths = distances.sum(dim=1)

    return route_lengths

def _reward_func(sample_solution):
    """
    Compute the reward for the routes as the negative value of the route lengths.

    Args:
        sample_solution (torch.Tensor): A tensor of shape [batch_size, decode_len, input_dim]
            representing the coordinates of each point in the sequence for each sample in the batch.

    Returns:
        torch.Tensor: Reward tensor of size [batch_size] containing the negative route length.
    """
    route_lengths = compute_route_length(sample_solution)

    rewards = -route_lengths
    
    return rewards

# # Example usage
# batch_size = 2
# decode_len = 3
# input_dim = 2
# sample_solution = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
#                                 [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])  # [batch_size, decode_len, input_dim]
# rewards = reward_func(sample_solution)
# print("Rewards:", rewards)
