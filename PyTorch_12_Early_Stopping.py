import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn,optim

import matplotlib.pyplot as plt
import numpy as np

class SampleDataset(Dataset):
    def __init__(self, train = True):
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        self.f = -3 * self.X + 1
        self.Y = self.f + 0.1 * torch.randn(self.X.size())
        if (train == True):
            self.X[50:] = 20
        else:
            pass
        self.len = self.X.shape[0]

    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
    
    def __len__(self):
        return self.len

training_dataset = SampleDataset()
validation_dataset = SampleDataset(train = False)

class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear = nn.Linear(in_features = input_size,out_features=output_size)

    def forward(self,x):
        return self.linear(x)
        
model = LR(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

trainloader = DataLoader(dataset = training_dataset,batch_size=1)

TRAINING_LOSS = []
VALIDATION_LOSS = []
ITERATIONS = []
min_loss = 100

def train_model_early_stopping(epochs,min_loss):
    iter = 1
    for epoch in range(epochs):
        for x,y in trainloader:
            ITERATIONS.append(iter)
            iter += 1
            y_pred = model(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss = criterion(model(training_dataset.X),training_dataset.Y).item()
            validation_loss = criterion(model(validation_dataset.X),validation_dataset.Y).item()
            TRAINING_LOSS.append(training_loss)
            VALIDATION_LOSS.append(validation_loss)

            if (validation_loss < min_loss ):
                value = epoch
                min_loss = validation_loss
                torch.save(model.state_dict(),'best_model.pt')

train_model_early_stopping(20,min_loss)

""" plt.plot(ITERATIONS,TRAINING_LOSS, label = 'Training cost')
plt.plot(ITERATIONS,VALIDATION_LOSS, label = 'Validation cost')
plt.xlabel("Iterations ")
plt.ylabel("Cost")
plt.legend(loc = 'upper right')
plt.show()
 """

print(ITERATIONS)

# Create a new linear regression model object
model_best = LR(1,1)
model_best.load_state_dict(torch.load('best_model.pt'))

plt.plot(model_best(validation_dataset.X).data.numpy(), label = 'best model')
plt.plot(model(validation_dataset.X).data.numpy(), label = 'maximum iterations')
plt.plot(validation_dataset.Y.numpy(), 'rx', label = 'true line')
plt.legend()
plt.show()


