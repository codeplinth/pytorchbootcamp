import torch
from torch import nn

torch.manual_seed(1)

#set the weight and bias
w = torch.tensor([[2.0],[3.0]],requires_grad=True)
b = torch.tensor([[1.0]],requires_grad=True)

#define the prediction function
def forward(x):
    y_pred = torch.mm(x,w) + b
    return y_pred

#calculate y_pred
x = torch.tensor([[1.0,2.0]])
y_pred = forward(x)
print(y_pred)

#Sample tensor X
X = torch.tensor([[1.0,1.0],[1.0,2.0],[1.0,3.0]])

#Make a prediction of X
y_pred = forward(X)
print(y_pred)

#Make a linear regression model using buitin function
model = nn.Linear(2,1)
print(list(model.parameters()))

#make a prediction of x
y_pred = model(x)
print(y_pred)

#make a prediciton of X
y_pred = model(X)
print(y_pred)

#create linear regression class
class MLR(nn.Module):
    #contructor
    def __init__(self,input_size,output_size):
        super(MLR,self).__init__()
        self.linear = nn.Linear(in_features = input_size, out_features = output_size)
    
    #prediction function
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = MLR(2,1)

#print model parameters
print(list(model.parameters()))

print(model.state_dict())

#make a prediction of x
y_pred = model(x)
print(y_pred)

#make a prediciton of X
y_pred = model(X)
print(y_pred)