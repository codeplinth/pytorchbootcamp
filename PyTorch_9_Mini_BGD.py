import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

class SampleData(Dataset):
    def __init__(self):
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        self.f = 1 * X - 1
        self.Y = self.f + 0.1 * torch.randn(self.X.shape[0])
        self.len = self.X.shape[0]
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
    def __len__(self):
        return self.len

def forward(x):
    return w * x + b

def criterion(y_pred,y):
    return torch.mean((y_pred - y) ** 2)

dataset = SampleData()
trainloader = DataLoader(dataset = dataset , batch_size= 1)

w = torch.tensor(-15.0 , requires_grad=True)
b = torch.tensor(-10.0 , requires_grad=True)
lr = 0.1
LOSS_SGD = []
EPOCH = []

def train_model_SGD(iter):
    for epoch in range(iter):
        LOSS_SGD.append(criterion(forward(X),Y).item())
        EPOCH.append(epoch)
        for x,y in trainloader:
            y_pred = forward(x)
            loss = criterion(y_pred,y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_SGD(10)

dataset = SampleData()
trainloader = DataLoader(dataset = dataset , batch_size= 5)

w = torch.tensor(-15.0 , requires_grad=True)
b = torch.tensor(-10.0 , requires_grad=True)
lr = 0.1
LOSS_BGD5 = []
EPOCH = []

def train_model_BGD5(iter):
    for epoch in range(iter):
        LOSS_BGD5.append(criterion(forward(X),Y).item())
        EPOCH.append(epoch)
        for x,y in trainloader:
            y_pred = forward(x)
            loss = criterion(y_pred,y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_BGD5(10)

dataset = SampleData()
trainloader = DataLoader(dataset = dataset , batch_size= 10)

w = torch.tensor(-15.0 , requires_grad=True)
b = torch.tensor(-10.0 , requires_grad=True)
lr = 0.1
LOSS_BGD10 = []
EPOCH = []

def train_model_BGD10(iter):
    for epoch in range(iter):
        LOSS_BGD10.append(criterion(forward(X),Y).item())
        EPOCH.append(epoch)
        for x,y in trainloader:
            y_pred = forward(x)
            loss = criterion(y_pred,y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_BGD10(10)

plt.plot(EPOCH,LOSS_SGD,label = 'Stochastic Gradient Descent')
plt.plot(EPOCH,LOSS_BGD5,label = 'Mini-Batch Gradient Descent Batch size - 5')
plt.plot(EPOCH,LOSS_BGD10,label = 'Mini-Batch Gradient Descent Batch size - 10')
plt.xlabel('Epoch')
plt.ylabel('LOSS')
plt.legend()
plt.show()