import torch
import torch.nn as nn
import sys

class Embedding():
    # This class is the base class for embedding the input graph.
    def __init__(self, emb_type, embedding_dim):
        self.emb_type = emb_type
        self.embedding_dim = embedding_dim
        
    def forward(self, input_pnt):
        # returns the embedded tensor. Should be implemented in child classes? idk what this means
        pass

class LinearEmbedding(Embedding):
    """ This class implements linear embedding.
    It is only a mapping to a higher dimensional space.
    """
    def __init__(self, embedding_dim):
        # Input: embedding_dim: embedding dimension
        super(LinearEmbedding, self).__init__('linear', embedding_dim)
        self.project_emb = nn.Conv1d(in_channels=2, out_channels=embedding_dim, kernel_size=1)
    
    def forward(self, input_pnt):
        # emb_inp_pnt: [batch_size, embedding_dim, max_time]
        a = input_pnt.transpose(1, 2)
        emb_inp_pnt = self.project_emb(input_pnt.transpose(1, 2))

        # emb_inp_pnt = tf.Print(emb_inp_pnt,[emb_inp_pnt])
        return emb_inp_pnt.transpose(1, 2)
    
if __name__ == "__main__":
    # Example usage
    # embedding_dim = 64
    embedding_dim = 128

    # Create LinearEmbedding instance
    linear_embedding = LinearEmbedding(embedding_dim)

    # Example input tensor
    # input_pnt = torch.randn(32, 10, 2)  # Batch size 32, max_time 10, input_dim 1
    input_pnt = torch.randn(2, 10, 2)  # Batch size 32, max_time 10, input_dim 1

    # Obtain embedded tensor
    emb_inp_pnt = linear_embedding.forward(input_pnt)

    # print("Embedded tensor shape:", emb_inp_pnt.shape)
    # Results: Embedded tensor shape: torch.Size([2, 10, 128])
    # Tested and verified behaviour