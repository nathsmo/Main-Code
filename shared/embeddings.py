import torch
import torch.nn as nn
import sys
import torch.nn.functional as F


class ConvEmbedding(nn.Module):
    """ This class implements linear embedding.
    It is only a mapping to a higher dimensional space.
    """
    def __init__(self, prt, embedding_dim):
        # Input: embedding_dim: embedding dimension
        super(ConvEmbedding, self).__init__()
        self.project_emb = nn.Conv1d(in_channels=2, out_channels=embedding_dim, kernel_size=1)
        prt.print_out("Embedding - linear")
    
    def forward(self, input_pnt):
        # emb_inp_pnt: [batch_size, embedding_dim, max_time]
        # print('Input pnt -in- forward Embedding: ', input_pnt.shape)
        input_pnt = input_pnt.float()  
        emb_inp_pnt = self.project_emb(input_pnt.transpose(1, 2))
        
        return emb_inp_pnt.transpose(1, 2)

'''
class EnhancedLinearEmbedding(nn.Module):
    def __init__(self, prt, input_dim, output_dim):
        super(EnhancedLinearEmbedding, self).__init__()
        prt.print_out("Embedding - enhanced")
        
        self.layer1 = nn.Conv1d(input_dim, 128, kernel_size=1)
        self.layer2 = nn.Conv1d(128, 256, kernel_size=1)
        self.layer3 = nn.Conv1d(256, output_dim, kernel_size=1)
        self.activation = nn.ReLU()
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # Ensure the feature map is reduced to 1

    def forward(self, x):
        x = x.transpose(1, 2).float()  # Assuming x is [batch_size, feature_dim, seq_len]
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x.transpose(1, 2)  # Transpose back to [batch_size, 1, output_dim]
'''

class EnhancedLinearEmbedding(nn.Module):
    """
    # Enhancements include:
    #     - Multiple convolution layers for hierarchical feature extraction.
    #     - Batch normalization after each convolution to stabilize learning.
    #     - Activation function between layers for non-linearity.
    #     - Residual connections to allow training of deeper networks.
    #     - Dropout for regularization to prevent overfitting.
    """
    def __init__(self, prt, num_channels, embedding_dim):
        super(EnhancedLinearEmbedding, self).__init__()
        prt.print_out("Embedding - enhanced 2")

        self.layer1 = nn.Conv1d(num_channels, embedding_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.activation1 = nn.ReLU()
        
        self.layer2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.activation2 = nn.ReLU()

        self.layer3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.activation3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)  # Dropout rate can be adjusted depending on the use case

        # Residual connection to adjust dimensions if necessary
        self.residual = nn.Conv1d(num_channels, embedding_dim, kernel_size=1)

    def forward(self, input_pnt):
        # Transpose input to match Conv1d expected input shape (batch, channels, sequence length)
        input_pnt = input_pnt.transpose(1, 2).float()

        # Pass through the first layer, activation, and batch normalization
        x = self.layer1(input_pnt)
        x = self.bn1(x)
        x = self.activation1(x)

        # Second convolutional layer, activation, and batch normalization
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation2(x)

        # Third convolutional layer, activation, and batch normalization
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.activation3(x)

        x = self.dropout(x)  # Apply dropout

        # Add the output of the last convolutional layer to the residual path output
        x += self.residual(input_pnt)

        # Transpose back to match expected output shape (batch, sequence length, channels)
        return x.transpose(1, 2)


class Enhanced__LinearEmbedding(nn.Module):
    """
    You can introduce non-linearity and additional complexity in several ways:

        Non-linear Activation Functions: 
            Adding an activation function like ReLU or LeakyReLU after the convolution can help the model learn more complex functions.
        Additional Convolution Layers: 
            Stacking multiple convolutional layers, possibly with different kernel sizes and dilation rates, can allow the 
            network to learn hierarchical features at different scales and contexts.
        Residual Connections: 
            Introducing skip connections between layers (like those used in ResNet architectures) can help in training deeper 
            networks by alleviating issues like vanishing gradients.
    """
    def __init__(self, prt, num_channels, embedding_dim):
        super(Enhanced__LinearEmbedding, self).__init__()
        # Initialize the first convolution layer with padding to maintain dimension
        prt.print_out("Embedding - enhanced og")
        self.layer1 = nn.Conv1d(num_channels, embedding_dim, kernel_size=3, padding=1)
        # ReLU activation
        self.activation = nn.ReLU()
        # Second convolution layer with padding to maintain dimension
        self.layer2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        # Residual connection to adjust dimensions if necessary
        self.residual = nn.Conv1d(num_channels, embedding_dim, kernel_size=1)

    def forward(self, input_pnt):
        # Transpose input to match Conv1d expected input shape (batch, channels, sequence length)
        input_pnt = input_pnt.transpose(1, 2).float()
        # Pass through first layer and activate
        x = self.activation(self.layer1(input_pnt))
        # Add the output of the second layer to the residual path output
        x = self.layer2(x) + self.residual(input_pnt)
        # Transpose back to match expected output shape (batch, sequence length, channels)
        return x.transpose(1, 2)

class MinimalLinearEmbedding(nn.Module):
    """ Implements a minimal linear embedding using a simple linear layer.
    It maps input data to a higher dimensional space linearly.
    """
    def __init__(self, prt, num_input_features, embedding_dim):
        super(MinimalLinearEmbedding, self).__init__()
        prt.print_out("Embedding - Minimal")
        self.num_input_features = num_input_features
        # A linear layer to project input features directly to the embedding dimension
        self.linear = nn.Linear(num_input_features, embedding_dim)

    def forward(self, input_pnt):
        # input_pnt expected shape: [batch_size, sequence_length, num_input_features]
        # Process each time step independently
        input_pnt = input_pnt.float()

        batch_size, seq_len, _ = input_pnt.shape
        # Flatten input to apply the linear layer, then reshape back to the sequence format
        input_pnt = input_pnt.reshape(-1, self.num_input_features)
        emb_inp_pnt = self.linear(input_pnt)
        emb_inp_pnt = emb_inp_pnt.reshape(batch_size, seq_len, -1)
        return emb_inp_pnt
    
# if __name__ == "__main__":
#     # Example usage
#     # embedding_dim = 64
#     embedding_dim = 128

#     # Create LinearEmbedding instance
#     # linear_embedding = LinearEmbedding(embedding_dim)
#     linear_embedding = EnhancedLinearEmbedding(2, embedding_dim)

#     # Example input tensor
#     # input_pnt = torch.randn(32, 10, 2)  # Batch size 32, max_time 10, input_dim 2
#     input_pnt = torch.randn(2, 10, 2)  # Batch size 2, max_time 10, input_dim 2

#     # Obtain embedded tensor
#     emb_inp_pnt = linear_embedding.forward(input_pnt)

#     # print("Embedded tensor shape:", emb_inp_pnt.shape)
#     # Results: Embedded tensor shape: torch.Size([2, 10, 128])
#     # Tested and verified behaviour