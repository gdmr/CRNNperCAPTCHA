import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image




class Dataset(Dataset):

    def __init__(self, img_dir):
        path_list = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir)

        self.img_list = [os.path.join(abspath, path) for path in path_list]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        

    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, idx):
        path = self.img_list[idx]

        label = os.path.basename(path).split('.')[0].lower().strip()
        img = Image.open(path).convert('RGB')

        img_tensor = self.transform(img)
        return img_tensor, label
