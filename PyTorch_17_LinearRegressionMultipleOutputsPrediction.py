import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

torch.manual_seed(1)


class Data2D(Dataset):
    def __init__(self):
        self.X = torch.zeros(20,2)
        self.X[:,0] = torch.arange(-1,1,0.1)
        self.X[:,1] = torch.arange(-1,1,0.1)
        self.w = torch.tensor([[1.0,-1.0],[1.0,3.0]])
        self.b = torch.tensor([1.0,-1.0])
        self.f = torch.mm(self.X,self.w) + self.b
        self.Y = self.f + 0.01 * torch.randn((self.X.shape[0],1))
        self.len = self.X.shape[0]
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
    def __len__(self):
        return self.len

dataset = Data2D()
trainloader = DataLoader(dataset=dataset , batch_size=5)

class LRMO(nn.Module):
    def __init__(self,input_size,output_size):
        super(LRMO,self).__init__()
        self.linear = nn.Linear(in_features = input_size, out_features = output_size)
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = LRMO(2,2,)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.1)
writer = SummaryWriter()

EPOCHS = 10
LOSS = []

def train_model(EPOCHS):
    for epoch in range(EPOCHS):
        for batch,data in enumerate(trainloader,0):
            x,y = data
            y_pred = model(x)
            loss = criterion(y_pred,y)
            LOSS.append(loss.item())
            writer.add_scalar("Loss/Epochs", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_model(10)
writer.close()
""" 
#plot using matplotlib
plt.plot(LOSS)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
 """

#tensorboard --logdir=runs