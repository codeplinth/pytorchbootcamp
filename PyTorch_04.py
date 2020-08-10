#Logistic Regression
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#input variable / features / independent variable
x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
#target variable / dependent variable
y_data = Variable(torch.Tensor([[0.0],[0.0],[1.0],[1.0]]))

#Step - 1 Model class
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()

#Step 2 - Construct Loss criterion and Optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

#Step 3 Training - Forward - Backward - Step
for epoch in range(400):
    #Forward pass : compute predicted y by passing x to the model
    y_pred = model(x_data)
    #Compute loss
    loss = criterion(y_pred,y_data)

    print(epoch,loss.data)
    #Zero gradients
    optimizer.zero_grad()
    #perform backward pass
    loss.backward()
    #update the weights
    optimizer.step()


hour_var = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour",1.0,model.forward(hour_var).data[0][0] > 0.5)
hour_var = Variable(torch.Tensor([[10.0]]))
print("predict 10 hour",10.0,model.forward(hour_var).data[0][0] > 0.5)

