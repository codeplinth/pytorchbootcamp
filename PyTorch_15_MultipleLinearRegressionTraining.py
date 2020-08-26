import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader

import matplotlib.pyplot as plt

#set the randome seed to 1
torch.manual_seed(1)

class Data2D(Dataset):
    #contructor
    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-1, 1, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b    
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]    
    #getter
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    #len
    def __len__(self):
        return self.len

#create dataset object
dataset = Data2D()

#create customized Multiple Linear Regression 
class MLR(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLR,self).__init__()
        self.linear = nn.Linear(in_features = input_size ,out_features = output_size)

    def forward(self,x):
        return self.linear(x)

model = MLR(2,1)
#print(list(model.parameters()))

#create the optimizer
optimizer = optim.SGD(model.parameters(),lr = 0.1)

#create the loss criterion
criterion = nn.MSELoss()

#create data loader
train_loader = DataLoader(dataset = dataset , batch_size = 2)

LOSS = []
EPOCHS = 10

def train_model(EPOCHS):
    for epoch in range(EPOCHS):
        for x,y in train_loader:
            y_pred = model(x)
            loss = criterion(y_pred,y)
            LOSS.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_model(10)

# Plot out the Loss and iteration diagram
plt.plot(LOSS)
plt.xlabel("Iterations ")
plt.ylabel("Cost/total loss ")   
plt.show()
