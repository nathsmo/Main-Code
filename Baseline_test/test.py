import numpy as np

filename_results = f"./data/vrp_test_1000_10_seed24601.npy"
data = np.load(filename_results)

first_matrix = data[0]
print(first_matrix)
print(first_matrix.shape)
# Rest of your code goes here