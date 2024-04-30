import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq models"""
    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh        # Set use_tanh flag
        self.C = C          # Set scaling factor C # tanh exploration parameter
        self.tanh = nn.Tanh() # Define hyperbolic tangent function
        
        """
        # Define the parameter with a specific size, here [1, dim]
        # Before it was supposed to get the variable v from somewhere else... but I just didn't know how to do that
        # self.v = nn.Parameter(torch.empty(1, dim))
        # Initialize it with Xavier initializer
        # init.xavier_uniform_(self.v)
        # Expand dimensions: now shape will be [1, dim, 1]
        # self.v = self.v.unsqueeze(2)
        # Replaced with: 
        self.v = nn.Parameter(torch.randn(1, dim, 1))
        """
        # Define linear layers for projection
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Linear(dim, dim)

        self.v = nn.Parameter(torch.randn(1, dim, 1))


    def forward(self, query, ref, *args, **kwargs): # called __call__ before
        """
        This function gets a query tensor and ref tensor and returns the logit op.
        Args: 
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder. 
                [batch_size x max_time x dim]

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # Project query tensor
        q = self.project_query(query)  # [batch_size x dim]

        # Apply convolution along the time dimension
        e = self.project_ref(ref)  # [batch_size x max_time x dim]
        """
        Comments from code before, and just in case it doesn't work.

        # expanded_q,e: [batch_size x max_time x dim]
        # expanded_q = q.unsqueeze(1).expand(-1, e.size(1), -1)  # [batch_size, max_time, dim]

        # Expand v to match the shape of e
        # v_view = self.v.expand(e.size(0), -1, -1)  # [batch_size x dim x 1]

        # Compute u: [batch_size x max_time x dim] * [batch_size x dim x 1] = [batch_size x max_time]
        # u = torch.matmul(self.tanh(expanded_q + e), v_view).squeeze(2)  
        """
        # Expand dimensions of q to match the shape of e
        expanded_q = q.unsqueeze(1) # [batch_size, max_time, dim]

        # Prepare v for batch multiplication
        v_view = self.v.expand(e.size(0), -1, -1)  # [batch_size x dim x 1]

        # Apply tanh activation function over the sum, prepare for multiplication
        tanh_output = self.tanh(expanded_q + e)
        
        # Compute the attention logits using batch matrix multiplication
        u = torch.bmm(tanh_output, v_view).squeeze(2)  # [batch_size x max_time]

        # Apply scaling factor and tanh if required
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u

        return e, logits
    
"""

Notes: 

Changes in project_ref from Conv to LinearNN
Semantic Equivalence: In the TensorFlow code, project_ref is used to project
    the reference tensor ref. However, the usage of tf.layers.Conv1D for this purpose
    might be misleading, as it's typically used for applying 1D convolutions to the 
    input. Instead, a more semantically appropriate choice would be a linear layer
    (nn.Linear in PyTorch) which performs a simple affine transformation without convolution.

Simplification and Consistency: Using nn.Linear for projection simplifies the code 
    and makes it consistent with project_query, which is also a linear projection. 
    This choice makes the purpose of project_ref clearer to anyone reading the code.

Flexibility: Although nn.Linear and nn.Conv1d can both perform linear 
    transformations, using nn.Linear offers more flexibility in terms of 
    configuration. For example, you can easily adjust the number of input and 
    output features in nn.Linear, whereas nn.Conv1d expects the input tensor to 
    have a specific shape suitable for convolution.
"""

"""
# Example usage
batch_size = 32
max_time = 10
dim = 64

# Create model instance
model = Attention(dim=dim)

# Create sample input tensors
query = torch.randn(batch_size, dim)
ref = torch.randn(batch_size, max_time, dim)

# Forward pass
e, logits = model(query, ref)

print("Convolved reference shape:", e.shape)
print("Logits shape:", logits.shape)
print("Logits:", logits)    
# Convolved reference shape: torch.Size([32, 10, 64])
# Logits shape: torch.Size([32, 10])
"""

#Tested and working as expected