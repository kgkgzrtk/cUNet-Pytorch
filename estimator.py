import argparse
import pickle
import os

import pandas as pd
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from utils import ImageLoader, FlickrDataLoader
from sampler import ImbalancedDatasetSampler

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str, default='')
parser.add_argument('--pkl_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='cp')
parser.add_argument('--name', type=str, default='noname-estimator')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()


#load data
df = pd.read_pickle(args.pkl_path)
print('{} data were loaded'.format(len(df)))

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(args.input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
            brightness=0.5,
            contrast=0.3,
            saturation=0.3,
            hue=0  
        ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
transform = {'train': train_transform, 'test': test_transform}
train_data_rate = 0.7
pivot = int(len(df) * train_data_rate)
df_shuffle = df.sample(frac=1)
df_sep = {'train': df_shuffle[:pivot], 'test': df_shuffle[pivot:]}
del df, df_shuffle
cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], cols, transform[s])

train_set = loader('train')
test_set = loader('test')

train_loader = torch.utils.data.DataLoader(
        train_set, 
        sampler=ImbalancedDatasetSampler(train_set),
        batch_size=args.batch_size, 
        num_workers=args.num_workers)

test_loader = torch.utils.data.DataLoader(
        train_set, 
        sampler=ImbalancedDatasetSampler(test_set),
        batch_size=args.batch_size, 
        num_workers=args.num_workers)

num_classes = train_set.num_classes

# modify exist resnet101 model
model = models.resnet101(pretrained=False, num_classes=num_classes)
model.cuda()

#train setting 
comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}'.format(args.lr, args.batch_size, args.num_epoch, args.input_size, args.name)
writer = SummaryWriter(comment=comment)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
criterion = nn.MSELoss()
eval_per_epoch = 1
save_per_epoch = 5
global_step = 0
tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
for epoch in tqdm_iter:
    loss_li = []
    for i, data in enumerate(train_loader, start=0):
        inputs, labels = (d.to('cuda') for d in data)
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        global_step += 1
        loss_li.append(loss.item())

        if epoch % eval_per_epoch == eval_per_epoch-1:
            loss_li_ = []
            for j, data_ in enumerate(test_loader, start=0):
                inputs_, labels_ = (d.to('cuda') for d in data_)
                outputs_ = model(inputs_)
                loss_ = criterion(outputs_, labels_)
                loss_li_.append(loss_.item())
            train_loss = np.mean(loss_li)
            test_loss = np.mean(loss_li_)
            writer.add_scalar('mse_loss/train', train_loss, global_step)
            writer.add_scalar('mse_loss/test', test_loss, global_step)
            tqdm_iter.set_description('{} iter: Train loss={:.5f} Test loss={:.5f}'.format(global_step, train_loss, test_loss))

    if epoch % save_per_epoch == 0:
        out_path = os.path.join(args.save_path, 'resnet101_'+str(epoch)+'.pt')
        os.makedirs(args.save_path, exist_ok=True) 
        torch.save(model, out_path)

print('Done: training')
