import os
import glob
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as TF
import random
from PIL import Image
from torchvision import io

    
class GPSDatasetSUP(Dataset):
    def __init__(self, metadata, root_dir, label_dir, train, transform=None, normalize=None):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.land_dir = label_dir
        self.transform = transform
        self.train = train
        self.normalize = normalize
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx].item())
        image_path = img_name + '.png'
        image = io.read_image(image_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        land_name = os.path.join(self.land_dir, self.metadata[idx].item())
        try:
            land_value = Image.open(land_name + '.tif')
            land_value = torch.tensor(np.array(land_value)).unsqueeze(0)
            land_value = torch.clamp(land_value, max=1)
            
        except:
            land_value = torch.zeros(1,256,256)
            
         #VFlip
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
        image, land_value = self.normalize(image, land_value)
        
        land_value = land_value.squeeze(0).long()
        

        return img_name, image, land_value
    
class GPSDatasetEVAL(Dataset):
    def __init__(self, metadata, root_dir, train, transform=None, normalize=None):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.normalize = normalize
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_name = self.metadata[idx].item()
        img_name = os.path.join(self.root_dir, image_name)
        image_path = img_name + '.png'
        image = io.read_image(image_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
            
        if image.shape[0] == 4:
            image = image[:3, :, :]

        land_value = torch.zeros(image.shape)
            
         #VFlip
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
        image, land_value = self.normalize(image, land_value)
        land_value = land_value.squeeze(0).long()
        

        return image_name, image, land_value