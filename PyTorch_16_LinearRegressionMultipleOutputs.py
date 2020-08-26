import torch
from torch import nn

torch.manual_seed(1)

class LRMO(nn.Module):
    def __init__(self,input_size,output_size):
        super(LRMO,self).__init__()
        self.linear = nn.Linear(in_features = input_size, out_features = output_size)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = LRMO(1,10)
print(list(model.parameters()))

x = torch.tensor([[1.0]])
y_pred = model(x)
print(y_pred)

X = torch.tensor([[1.0],[2.0],[3.0]])
y_pred = model(X)
print(y_pred)
