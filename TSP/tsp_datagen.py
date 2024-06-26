import numpy as np
import os
import torch

class DataGenerator():
    def __init__(self, args):
        """
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test (n_problems)
                args['n_nodes']: number of nodes in the problem. (ex.:locations)
                args['batch_size']: batch size for training
        """
        self.args = args
        self.data_dir = './data'
        self.n_problems = args['test_size']
        
        self.reset()
    
    def reset(self):
        """
        Reset the data pointer
        """
        self.count = 0

    def get_train_next(self):
        """
        Gets the next batch of training data for the TSP
        Returns:
            input_data: a random torch tensor with shape [batch_size, n_nodes, 2] so for tsp with batch_size=128, n_nodes=10, 2D coordinates
        """
        input_data = torch.rand(self.args['batch_size'], self.args['n_nodes'], 2)
        
        return input_data

    def get_train_data(self):
        """
        Get the train data. 
        Returns:
            
        """
        return self.create_dataset(self.args['batch_size'], self.args['n_nodes'], self.data_dir, seed=self.args['random_seed'], data_type='train')
    

    def get_test_data(self):
        """
        Get the test data
        """
        return self.create_dataset(self.n_problems, self.args['n_nodes'], self.data_dir, seed=self.args['random_seed'], data_type='test')
    
    def create_dataset(self, n_problems, n_nodes, data_dir, seed=None, data_type='train'):
        """
        This function creates TSP instances and saves them on disk. If a file is already available,
        it will load the file.
        Input:
            n_problems: number of problems to generate. (scenarios/data sets)
            n_nodes: number of nodes in the problem. (ex.:locations)
            data_dir: the directory to save or load the file.
            seed: random seed for generating the data.
            data_type: the purpose for generating the data. It can be 'test', 'train', 'val', or any string.
        output:
            data: a numpy array with shape [n_problems x n_nodes x 2]

        """

        # Define the filename based on input parameters
        # Include seed in filename only if it is not None
        filename = f"vrp_{data_type}_{n_problems}_{n_nodes}_seed{seed if seed is not None else 'random'}.npy"
        filepath = os.path.join(data_dir, filename)
        
        # Check if the file exists
        if os.path.exists(filepath):
            # print('Loading dataset for {}...'.format(filename))
            # Load the file if it already exists
            data = np.load(filepath)
        else:
            # print('Creating dataset for {}...'.format(filename))
            # Set the random seed only if it is provided
            if seed is not None:
                np.random.seed(seed)
            
            # Generate the data # Create an array for x, y coordinates and demands
            # Shape: [n_problems, n_nodes, 2]
            data = torch.rand(n_problems, n_nodes, 2)
            
            if data_type == 'test': 
                # Save the generated data to a file
                np.save(filepath, data)
                print('Data saved to {}...'.format(filepath))
        
        # print(data_type, ' <- data created with shape: ', data.shape)

        return data


"""
# Use this test code to check the implementation
# this should replace the get_test_next function previous here.

test_loader = DataLoader(self.test_dataset.get_test_data(), batch_size=self.args['batch_size']) # although the batch size is used for training...

for num_test in test_loader:
    # Processing each test batch
    # Do something with the batch
    print(batch.shape)
"""
