import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn,optim
import numpy as np

import matplotlib.pyplot as plt

import sys

class SampleDataset(Dataset):
    def __init__(self,train = True):
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        self.f = -3 * self.X + 1
        self.Y = self.f + 0.1 * torch.randn(self.X.size())
        self.len = self.X.shape[0]

        #outliers
        if (train == True ):
            self.Y[0] = 0
            self.Y[50:55] = 20
        else: 
            pass

    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]

    def __len__(self):
        return self.len

train_dataset = SampleDataset()
validation_dataset = SampleDataset(train = False)

class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear = nn.Linear(in_features=input_size,out_features=output_size)

    def forward(self,x):
        return self.linear(x)

trainloader = DataLoader(dataset = train_dataset , batch_size = 1)
criterion = nn.MSELoss()

# Create Learning Rate list, the error lists and the MODELS list
learning_rates = [0.0001,0.001,0.01,0.1]
train_error = np.zeros(len(learning_rates))
validation_error = np.zeros(len(learning_rates))
MODELS = []

def train_model_with_lr(iter,lr_list):
    for i,lr in enumerate(lr_list):
        model = LR(1,1)
        optimizer = optim.SGD(model.parameters(), lr = lr)
        for epoch in range(iter):
            for x,y in trainloader:
                y_pred = model(x)
                loss = criterion(y_pred,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_train = model(train_dataset.X)
        training_loss = criterion(y_train,train_dataset.Y)
        train_error[i] = training_loss.item()

        y_val = model(validation_dataset.X)
        validation_loss = criterion(y_val,validation_dataset.Y)
        validation_error[i] = validation_loss.item()

        MODELS.append(model)


train_model_with_lr(10,learning_rates)


plt.semilogx(learning_rates, train_error, label = 'Training Loss')
plt.semilogx(learning_rates, validation_error, label = 'Validation Loss')
plt.ylabel('Total Loss')
plt.xlabel('Learning Rate')
plt.legend()
plt.show()

i = 0
for model,learning_rate in zip(MODELS,learning_rates):
    y_val = model(validation_dataset.X)
    plt.plot(validation_dataset.X.numpy(),y_val.detach().numpy(),label = 'lr : ' + str(learning_rate))
plt.plot(validation_dataset.X.numpy(),validation_dataset.Y.numpy(),'or',label = 'Validation data')    
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


