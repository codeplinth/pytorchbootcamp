import torch
from torch.nn import Linear,Module

""" 
# Define w = 2 and b = -1 for y = wx + b
w = torch.tensor(2.0,requires_grad = True)
b = torch.tensor(-1.0,requires_grad = True)

# Function forward(x) for prediction
def forward(x):
    y_pred = w * x + b
    return y_pred

# Predict y = 2x - 1 at x = 1
x = torch.tensor([[1.0]])
y_pred = forward(x)
print('Prediction : {}'.format(y_pred))

# Create x Tensor and check the shape of x tensor

x = torch.tensor([[1.0], [2.0]])
print('The shape of x: ', x.shape)

# Make the prediction of y = 2x - 1 at x = [1, 2]

y_pred = forward(x)
print('Prediction : {}'.format(y_pred))

#Make a prediction of y = 2x - 1 at x = [[1.0], [2.0], [3.0]]
x = torch.tensor([[1.0], [2.0], [3.0]])
y_pred = forward(x)
print('Prediction : {}'.format(y_pred))

torch.manual_seed(1)
# Create Linear Regression Model, and print out the parameters

model = Linear(in_features = 1, out_features = 1,bias = True)
print('Parameters w & b are {}'.format(list(model.parameters())))
print(model.state_dict())
print(model.state_dict().keys())
print(model.state_dict().values())
print(model.weight)
print(model.bias)

# Make the prediction at x = [[1.0]]
x = torch.tensor([[1.0]])
y_pred = model(x)
print('The prediction: {}'.format(y_pred))

# Make the prediction using linear model at x = [[1.0], [2.0]]

x = torch.tensor([[1.0], [2.0]])
y_pred = model(x)
print('The prediction: {}'.format(y_pred))
 """

class LR(Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear = Linear(input_size,output_size)
    def forward(self,x):
        return self.linear(x)

model = LR(1,1)

print('Parameters are : {}'.format(list(model.parameters())))
print('Model is : {}'.format(model))

# Make the prediction using linear model at x = [[1.0], [2.0]]

x = torch.tensor([[1.0], [2.0]])
y_pred = model(x)
print('The prediction: {}'.format(y_pred))  
