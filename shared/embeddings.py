import torch
import torch.nn as nn
import sys

class LinearEmbedding(nn.Module):
    """ This class implements linear embedding.
    It is only a mapping to a higher dimensional space.
    """
    def __init__(self, prt, embedding_dim):
        # Input: embedding_dim: embedding dimension
        super(LinearEmbedding, self).__init__()
        self.project_emb = nn.Conv1d(in_channels=2, out_channels=embedding_dim, kernel_size=1)
        prt.print_out("Embedding - linear")
    
    def forward(self, input_pnt):
        # emb_inp_pnt: [batch_size, embedding_dim, max_time]
        # print('Input pnt -in- forward Embedding: ', input_pnt.shape)
        input_pnt = input_pnt.float()  
        emb_inp_pnt = self.project_emb(input_pnt.transpose(1, 2))
        
        return emb_inp_pnt.transpose(1, 2)

class EnhancedLinearEmbedding(nn.Module):
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
        super(EnhancedLinearEmbedding, self).__init__()
        # Initialize the first convolution layer with padding to maintain dimension
        prt.print_out("Embedding - enhanced")
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