import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

data_dir = ''
csv_file = 'index.csv'
""" 
csv_path = os.path.join(data_dir,csv_file)

data_name = pd.read_csv(csv_path)
print(data_name.head(5))
print('File name : {}'.format(data_name.iloc[0,1]))
print('y : {}'.format(data_name.iloc[0,0]))
print('Total rows : {}'.format(data_name.shape[0]))

image_name = data_name.iloc[0,1]
image_path = os.path.join(data_dir,image_name)

image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[0, 0])
plt.show() 
"""
class ImageDataset(Dataset):
    def __init__(self,data_dir,csv_file,transform = None):
        self.data_dir = data_dir
        csv_path = os.path.join(self.data_dir,csv_file)
        self.data_name = pd.read_csv(csv_path)
        self.len = self.data_name.shape[0]
        self.transform = transform
        
    def __getitem__(self,idx):
        image_path = os.path.join(self.data_dir, self.data_name.iloc[idx,1])
        image = Image.open(image_path)
        y = self.data_name.iloc[idx,0]
        if self.transform:
            image =  self.transform(image)
        return image,y

    def __len__(self):
        return self.len

""" 
my_image_dataset = ImageDataset(data_dir = data_dir,csv_file = csv_file)

image,y = my_image_dataset[0]
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show() 
"""

data_transform = transforms.Compose([transforms.CenterCrop(10), transforms.ToTensor()])
my_image_dataset = ImageDataset(data_dir = data_dir,csv_file = csv_file,transform=data_transform)

#my_image_dataset = ImageDataset(data_dir = data_dir,csv_file = csv_file,transform=None)

#print(my_image_dataset[0][0].shape)

image,y = my_image_dataset[1]
print(y)
plt.imshow(transforms.ToPILImage()(image),cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show() 