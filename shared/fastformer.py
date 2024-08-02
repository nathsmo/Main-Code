# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fast_transformer_pytorch import FastTransformer

# # Define the model
# model = FastTransformer(
#     num_tokens=20000,  # This should match the vocabulary size, adjust if necessary
#     dim=512,
#     depth=2,
#     max_seq_len=4096,
#     absolute_pos_emb=True
# )
# # Example coordinates tensor (64 problems, each with 10 addresses of 2 coordinates each)
# original_coordinates = torch.randn(64, 10, 2)  # Shape: (batch_size, num_addresses, num_coordinates)


# def fastformer_process(original_coordinates):
#     # Normalize the coordinates to fit into a range of token sIDs
#     # normalized_coordinates = (original_coordinates + 1) / 2  # Now in range [0, 1]
#     # original_coordinates should already be in range of [0, 1]
#     # Convert normalized coordinates to token IDs
#     token_ids_x = torch.clamp((original_coordinates[:, :, 0] * 19999).long(), 0, 19999)
#     token_ids_y = torch.clamp((original_coordinates[:, :, 1] * 19999).long(), 0, 19999)
#     # Interleave x and y token IDs to form a sequence
#     token_ids = torch.stack((token_ids_x, token_ids_y), dim=2).view(64, -1)  # Shape: (64, 20)
#     # Pad the token IDs to fit the max_seq_len expected by the transformer
#     padded_token_ids = nn.functional.pad(token_ids, (0, 4096 - token_ids.size(1)), 'constant', 0)  # Shape: (64, 4096)
#     # Create a mask for the sequence
#     mask = torch.zeros(64, 4096).bool()  # Initialize mask with zeros
#     mask[:, :20] = 1  # Set the valid positions to 1
#     # Get action logits from the transformer
#     logits = model(padded_token_ids, mask=mask)  # Shape: (64, 4096, 20000)
#     # Extract the logits corresponding to the original 10 positions
#     action_logits = logits[:, :20:2, :]  # Shape: (64, 10, 20000)
#     # Create a tensor to map the selected indices to the original coordinates
#     index_tensor = torch.arange(10).unsqueeze(0).unsqueeze(-1).expand(64, -1, 2).to(original_coordinates.device)
#     # Print the shape of action_logits
#     return action_logits, index_tensor

# def select_action_greedy(action_logits):
#     """
#     Selects actions using greedy method (highest logit value).
#     Args:
#         action_logits (torch.Tensor): The action logits of shape (batch_size, num_actions, num_tokens).
#     Returns:
#         selected_indices (torch.Tensor): The indices of selected actions of shape (batch_size, num_actions).
#     """
#     # Select the action with the highest logit value
#     selected_token_indices = torch.argmax(action_logits, dim=-1)
#     selected_coordinate_indices = torch.argmax(action_logits, dim=1)
#     return selected_coordinate_indices

# def remove_duplicates_preserve_order(tensor_row):
#     # Create a mask to keep track of unique elements
#     seen = set()
#     mask = []
#     for elem in tensor_row:
#         if elem.item() not in seen:
#             seen.add(elem.item())
#             mask.append(True)
#         else:
#             mask.append(False)
#     return tensor_row[torch.tensor(mask, dtype=torch.bool)]

# def obtain_original_coordinates(original_coordinates, unique_coord, batch_size=64, decoder_len=10):
#     # Initialize the result tensor
#     result = torch.zeros((batch_size, decoder_len, 2), dtype=original_coordinates.dtype)

#     # Iterate over each batch to gather the coordinates based on indices in tensor_2
#     for i in range(original_coordinates.size(0)):
#         result[i] = original_coordinates[i, unique_coord[i]]
#     return result

# action_logits, index_tensor = fastformer_process(original_coordinates)

# # Select actions using greedy method
# selected_coordinate_indices_greedy = select_action_greedy(action_logits)
# print("Selected coordinate indices (greedy):", selected_coordinate_indices_greedy)
# print("Selected coordinate indices (greedy):", selected_coordinate_indices_greedy.size())

# # Apply the function to each row of the tensor
# result = torch.stack([remove_duplicates_preserve_order(row) for row in selected_coordinate_indices_greedy])
# print("Selected unique coordinate indices (greedy):", result)
# print("Selected unique coordinate indices (greedy):", result.size())

# selected_coordinates_greedy = obtain_original_coordinates(original_coordinates, result, batch_size=64, decoder_len=10)
# print("Selected coordinates (greedy):", selected_coordinates_greedy)



