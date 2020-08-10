import torch
from torch.autograd import Variable

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

lr = 1e-2

w = Variable(torch.Tensor([1.0]),requires_grad=True)
print(w)

#model
def forward(x):
    return x * w

#loss
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

#before training
print("predict (before training)",4,forward(4).data[0])

#training loop
for epoch in range(10):
    for x_val,y_val in zip(x_data,y_data):
        l = loss(x_val,y_val)
        l.backward()
        print("\tgrad: ",x_val,y_val,w.grad.data[0])
        w.data = w.data - lr * w.grad.data

        #manually zero the gradients after updating weights
        w.grad.data.zero_()
    print("progress: ",epoch,"w=",w.data[0],"loss=",l.data[0])

#after training
print("predict (after training)",4,forward(4).data[0])

