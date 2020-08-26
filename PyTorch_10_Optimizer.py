import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn,optim
from torch.utils.tensorboard import SummaryWriter

import sys

class SampleDataset(Dataset):
    def __init__(self):
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        self.f = 1 * self.X - 1
        self.Y = self.f + 0.1 * torch.randn(self.X.size())
        self.len = self.X.shape[0]
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
    def __len__(self):
        return self.len

class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear = nn.Linear(in_features = input_size,out_features = output_size)

    def forward(self,x):
        return self.linear(x)

model = LR(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.1)

dataset = SampleDataset()
trainloader = DataLoader(dataset = dataset , batch_size = 1)

# Customize the weight and bias by overriding the parameters initialzed by pytorch

model.state_dict()['linear.weight'][0] = -15.0
model.state_dict()['linear.bias'][0] = -10.0

writer = SummaryWriter()

def train_model_SGD(iter):
    for epoch in range(iter):
        for x,y in trainloader:
            y_pred = model(x)
            loss = criterion(y_pred,y)
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_model_SGD(10)


writer.close()
sys.exit()
print(model.state_dict())




