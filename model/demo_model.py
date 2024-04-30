import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 10 input features, 5 output features

    def forward(self, x):
        return self.fc1(x)

# Create an instance of the network
net = SimpleNet()

# Define an optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Define a simple loss function
loss_function = nn.MSELoss()

# Sample data: input and target output
input_data = torch.randn(10, 10)  # batch of 10, 10 features each
target_data = torch.randn(10, 5)  # batch of 10, 5 target features each

# Training loop
for epoch in range(5):  # loop over the dataset multiple times
    optimizer.zero_grad()   # zero the parameter gradients
    outputs = net(input_data)  # forward pass
    loss = loss_function(outputs, target_data)  # calculate loss
    loss.backward()  # backward pass
    optimizer.step()  # update parameters
    
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# After this loop, `net` retains updated weights and can be used for further processing or evaluation
