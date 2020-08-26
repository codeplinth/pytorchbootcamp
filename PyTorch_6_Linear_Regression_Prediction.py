import torch
import matplotlib.pyplot as plt

# Create the f(X) with a slope of -3
X = torch.arange(-3,3,0.1).view(-1,1)
f = -3 * X

# Plot the line with blue

""" plt.plot(X.numpy(),f.numpy(),label = 'f')
plt.xlabel('X')
plt.ylabel('f')
plt.legend()
plt.show()
 """

# Add some noise to f(X) and save it in Y
Y = f + 0.1 * torch.randn(X.size())

# Plot the data points
""" 
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')

plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
 """

def forward(x):
    return w * x

def criterion(y_pred,y):
    return torch.mean((y_pred - y) ** 2)

lr = 0.1
LOSS = []
EPOCH = []

#Define a function to train the model
def train_model(iter):
    for epoch in range(iter):
        EPOCH.append(epoch)
        #calculate predicted y
        y_pred = forward(X)

        #calculate loss
        loss = criterion(y_pred,Y)

        LOSS.append(loss.item())

        #compute gradient of loss wrt all learnable parameters
        loss.backward()

        #update parameters

        w.data = w.data - lr * w.grad.data

        #zero the gradients before running the next back ward pass
        w.grad.data.zero_()

w = torch.tensor(-10.0,requires_grad=True)

train_model(4)

plt.plot(EPOCH,LOSS)
plt.show()