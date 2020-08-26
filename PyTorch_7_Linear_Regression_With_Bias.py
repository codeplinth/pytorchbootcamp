import torch
import matplotlib.pyplot as plt
import numpy as np

# create f(X) with a slope of 1 and bias of -1
X = torch.arange(-3,3,0.1).view(-1,1)
f = 1 * X - 1

#Add noise
Y = f + 0.1 * torch.randn(X.size())

""" 
# Plot out the line and the points with noise
plt.plot(X.numpy(),f.numpy(),label = 'f')
plt.plot(X.numpy(),Y.numpy(),'rx',label = 'Y')
plt.xlabel('X')
plt.ylabel('f/Y')
plt.legend()
plt.show()
"""

#Define the forward function
def forward(x):
    return w * x + b

#Define the MSE function
def criterion(y_pred,y):
    return torch.mean((y_pred - y) ** 2)

# Define the parameters w, b for y = wx + b
w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)

#Define learning rate and create an empty list to store the loss for each iteration
lr = 0.1
LOSS = []
EPOCH = []

#Define the function for training the model
def train_model(iter):
    for epoch in range(iter):
        #Make a prediction
        y_pred = forward(X)

        #calculate loss
        loss = criterion(y_pred,Y)

        #append loss to LOSS
        LOSS.append(loss.item())
        EPOCH.append(epoch)

        #compute the gradients wrt all learnable parameters
        loss.backward()

        #update the parameters
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data

        #zero the gradients before next backward pass
        w.grad.data.zero_()
        b.grad.data.zero_()

train_model(15)

plt.plot(EPOCH,LOSS)
plt.xlabel('Epoch')
plt.ylabel('LOSS')
plt.show()
