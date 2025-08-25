import argparse
from dataloader import GPSDatasetSUP
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model.UNet.UNet import UNet
import os
import numpy as np
import random
from augmentation import *
    
    
class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().tolist(), miou

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().tolist(), mf1
    
    def compute_precision(self):
        precision = self.hist.diag() / self.hist.sum(0)
        mp = precision[~precision.isnan()].mean().item()
        return precision.cpu().numpy().tolist(), mp
        
    def compute_recall(self):
        recall = self.hist.diag() / self.hist.sum(1)
        mrecall = recall[~recall.isnan()].mean().item()
        return recall.cpu().numpy().tolist(), mrecall
    
    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().tolist(), macc
     
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, pred, target):
        CE = F.cross_entropy(pred, target, reduction='none', ignore_index=255)
        pt = torch.exp(-CE)
        loss = ((1 - pt) ** 2) * CE # gamma
        alpha = torch.Tensor([args.weight, 1-args.weight]) # alpha(bigger for 1(pos))
        alpha = (target==0) * alpha[0] + (target==1) * alpha[1]
        return torch.mean(alpha * loss)

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


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

set_seed(0)

parser = argparse.ArgumentParser(description='GerDetection Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='training epoch')
parser.add_argument('--weight', default=0.1, type=float, help='focal loss weight')
args = parser.parse_args()        

global_step = 0
best_miou = 0

traintransform = get_train_augmentation([256, 256], 255)
valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()


labelset = GPSDatasetSUP(metadata=f'./multiyear/metadata/MNG_Ulaanbaatar_train_metadata.csv', 
                      root_dir=f'./multiyear/MNG_Ulaanbaatar/image',
                      label_dir=f'./multiyear/MNG_Ulaanbaatar/label',
                      train = True,
                      transform=traintransform,
                      normalize=normalize)


testset = GPSDatasetSUP(metadata=f'./multiyear/metadata/MNG_Ulaanbaatar_train_metadata.csv', 
                      root_dir=f'./multiyear/MNG_Ulaanbaatar/image',
                      label_dir=f'./multiyear/MNG_Ulaanbaatar/label',
                      train = False,
                      transform=valtransform,
                      normalize=normalize)


trainloader = torch.utils.data.DataLoader(labelset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

device = 'cuda' 
class_num = 2

net = UNet(backbone='resnet50', nclass=2)

if device == 'cuda':
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

net.to(device)

# Optimizer
optimizer = torch.optim.SGD(net.module.parameters(), lr=0.01, momentum=0.99)    
criterion = FocalLoss().cuda()
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

def train(epoch):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    
    for batch_idx, (names, inputs, targets) in enumerate(trainloader):
            
        batch_size = inputs.shape[0]
        inputs =  inputs.cuda()
        targets =  targets.long().cuda()
        
        outputs  = net(inputs)
        
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        
        if batch_idx % 100 == 0:
            print("[BATCH_IDX : {}, LOSS : {}]".format(batch_idx, loss) )
        optimizer.step()
        
        global_step += 1
    scheduler.step()
        

def test(epoch):
    global best_miou
    net.eval()
    correct = 0
    
    metrics = Metrics(2, 255)
    
    
    for batch_idx, (names, images, targets) in enumerate(testloader):
        inputs =  images.to(device)
        targets =  targets.long().to(device)
        outputs = net(inputs)
        outputs = F.softmax(outputs, dim=1)
        metrics.update(outputs.cpu(), targets.cpu())
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    precision, mprecision = metrics.compute_precision()
    recall, mrecall = metrics.compute_recall()
    
    print("f1 : ", f1)
    print("miou : ", ious)
    print("Pixel Acc : ", acc)
    print("best_mf1 : ", best_f1)
    print("Recall: ", recall)
    print("precision: ", precision)

    
    
    if miou >= best_miou:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': miou,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/UNet_MNG_{args.weight}.t7')
        best_f1 = miou
        
            

for epoch in range(0, args.epoch):
    train(epoch)
    test(epoch)