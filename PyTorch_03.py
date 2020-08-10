#Linear Regression
import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.optim as optim

#input variable / features / independent variable
x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
#target variable / dependent variable
y_data = Variable(torch.Tensor([[2.0],[4.0],[6.0]]))

#Step - 1 Model class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() 
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

#Step 2 - Construct Loss criterion and Optimizer

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

#Step 3 Training - Forward - Backward - Step
for epoch in range(500):
    #Forward pass : compute predicted y by passing x to the model
    y_pred = model(x_data)

    #Compute loss
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    #Zero gradients
    optimizer.zero_grad()

    #perform backward pass
    loss.backward()

    #update the weights
    optimizer.step()
    
hour_var = Variable(torch.Tensor([[4.0]]))
print("Prediction after training",4,model.forward(hour_var).data[0][0])
