#Deep and wide model
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np


xy = np.loadtxt('diabetes.csv',delimiter=',',dtype=np.float32)
# pylint: disable=E1101
x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])
# pylint: enable=E1101

#Step 1 Model Class
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

#Step 2 Construct loss criterion and Optimizer

criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(),lr = 0.01)

#Step 3 Training
for epoch in range(100):
    y_pred = model(x_data)

    loss = criterion(y_pred,y_data)

    print(epoch+1,loss.item())
    #Zero gradients
    optimizer.zero_grad()
    
    loss.backward()

    optimizer.step()
