import argparse
from dataloader import GPSDatasetEVAL
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from model.UNet.UNet import UNet
from torch.autograd import Variable
import os
import numpy as np
import random
from torch.utils.data import Subset
from augmentation import *
import copy 
import glob
import pandas as pd
from PIL import Image
import tifffile

        
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    
set_seed(0)

parser = argparse.ArgumentParser(description='GerDetection Evaluation')
parser.add_argument('--model', type=str, help='model path')

args = parser.parse_args()

best_f1 = 0
first_flag = True
target_dict = {}


def get_train_augmentation(size, seg_fill):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_val_augmentation(size):
    return Compose([
        Resize(size),
    ])


model_name = args.model.split("/")[-1].split(".")[0]
train_timeline = model_name.split("_")[-1]
dir_path = f'./result/UNet_{args.city}_binary'

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
    

global_step = 0

traintransform = get_train_augmentation([256, 256], 255)
strongtransform = get_strong_augmentation([256, 256], 255)
valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()


testset = GPSDatasetEVAL(metadata=f"./multiyear/metadata/MNG_eval_metadata.csv",
                      root_dir=f'./unlabeled/MNG_Ulaanbaatar',
                      train = False,
                      transform=valtransform,
                      normalize=normalize)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

device = 'cuda' 
class_num = 2

net = UNet(backbone='resnet50', nclass=2)

if device == 'cuda':
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(args.model)['net'], strict = True)
    cudnn.benchmark = True
    
net.to(device)

def eval_image(epoch):
    net.eval()    
    
    for batch_idx, (names, images, _) in enumerate(testloader):
        inputs =  images.to(device)
        outputs = net(inputs)
        outputs = torch.argmax(outputs, dim=1)
        for name, output in zip(names, outputs):
            out = output.cpu().detach().numpy().astype(np.uint8)
            if not np.array_equal(out, np.zeros(out.shape)):
                img = Image.fromarray(out)

                year = name.split("/")[0]
                if not os.path.isdir(os.path.join(dir_path, year)):
                    os.mkdir(os.path.join(dir_path, year))
                
                img.save(os.path.join(dir_path, name+".tif"), "TIFF")
    
            

eval_image(0)