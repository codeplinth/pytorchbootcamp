import torch
import matplotlib.pyplot as plt

#create some sample data
X = torch.arange(-3,3,0.1).view(-1,1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

""" 
#plot the data and the line
plt.plot(X.numpy(),f.numpy(),label = 'f')
plt.plot(X.numpy(),Y.numpy(),'ro',label = 'Y')
plt.xlabel('X')
plt.ylabel('f/Y')
plt.legend()
plt.show()

"""

def forward(x):
    return w * x + b

def criterion(y_pred,y):
    return torch.mean((y_pred - y) ** 2)

w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)

lr = 0.1
LOSS_BGD = []
EPOCH = []

def train_model_BGD(iter):
    for epoch in range(iter):
        y_pred = forward(X)
        loss = criterion(y_pred,Y)
        LOSS_BGD.append(loss.item())
        EPOCH.append(epoch)
        loss.backward()
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

train_model_BGD(10)

w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)

lr = 0.1
LOSS_SGD = []
EPOCH = []

def train_model_SGD(iter):
    for epoch in range(iter):
        LOSS_SGD.append(criterion(forward(X),Y).item())
        EPOCH.append(epoch)
        for x,y in zip(X,Y):
            y_pred = forward(x)
            loss = criterion(y_pred,y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_SGD(10)

plt.plot(LOSS_BGD,EPOCH,label='BGD')
plt.plot(LOSS_SGD,EPOCH,label='SGD')
plt.xlabel('Epoch')
plt.ylabel('LOSS')
plt.legend()
plt.show()

