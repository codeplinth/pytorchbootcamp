import torch
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np

class DiabetesDataset(Dataset):
    #initialize data
    def __init__(self):
        xy = np.loadtxt('diabetes.csv',delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        # pylint: disable=E1101
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        # pylint: enable=E1101
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

dataset = DiabetesDataset()

train_loader = DataLoader(dataset = dataset,
                            batch_size=32,
                            shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1 = nn.Linear(8,6)
        self.l2 = nn.Linear(6,4)
        self.l3 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model = Model()
#print(model)

criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(40):
    for batch_idx,data in enumerate(train_loader,0):
        #get inputs
        x_data,y_data = data
        y_pred = model(x_data)
        loss = criterion(y_pred,y_data)
        print("Epoch : {} Batch : {} Loss : {}".format(epoch+1,batch_idx+1,loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

