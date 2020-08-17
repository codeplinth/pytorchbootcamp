import torch
from torch.utils.data import Dataset

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

our_dataset = toy_set()

for idx in range(3):
    x,y = our_dataset[idx]
    print("x = {} y = {}".format(x,y))

for x,y in our_dataset:
    print("x = {} y = {}".format(x,y))
