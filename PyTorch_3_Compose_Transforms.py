import torch
from torch.utils.data import Dataset
from torchvision import transforms

#crate a subclass from Dataset
class toy_set(Dataset):
    #contructor
    def __init__(self,length = 50,transform = None):
        self.len = length
        # pylint: disable=E1101 
        self.x = torch.ones(length,2)
        self.y = torch.ones(length,1)
        # pylint: enable=E1101
        self.transform = transform

    #return data at a given index
    def __getitem__(self,index):
        sample = self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    #return length
    def __len__(self):
        return self.len

#crate a transforms class
class add_mult(object):
    #contructor
    def __init__(self,addx = 1,muly = 2):
        self.addx = addx
        self.muly = muly
    #executor
    def __call__(self,sample):
        return sample[0] + self.addx , sample[1] * self.muly

class mult(object):
    def __init__(self,mul = 100):
        self.mult = mul
    def __call__(self,sample):
        return sample[0] * self.mult , sample[1] * self.mult

# Combine the mult() & add_mult()
data_transform = transforms.Compose([mult(),add_mult()])

#print(data_transform)
data_set = toy_set()
composed_data_set = toy_set(transform = data_transform)

for idx in range(3):
    x,y = data_set[idx]
    print("i = {} Original x = {} Original y = {}".format(idx,x,y))
    x_c,y_c = composed_data_set[idx]
    print("i = {} Transformed x = {} Transformed y = {}".format(idx,x_c,y_c))
