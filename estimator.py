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

from utils import ImageLoader, FlickrDataLoader
from sampler import ImbalancedDatasetSampler

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str, default='')
parser.add_argument('--pkl_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='cp')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)

args = parser.parse_args()


def precision(outputs, labels):
    a = torch.argmax(outputs, dim=1)
    one_hot = nn.functional.one_hot(labels, num_classes=num_classes)
    b = torch.argmax(one_hot, dim=1)
    return torch.eq(a, b).float().mean()

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
cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], cols, transform[s])

train_set = loader('train')
test_set = loader('test')

train_loader = torch.utils.data.DataLoader(
        train_set, 
        sampler=ImbalancedDatasetSampler(train_set),
        batch_size=args.batch_size, 
        num_workers=8)

test_loader = torch.utils.data.DataLoader(
        train_set, 
        sampler=ImbalancedDatasetSampler(test_set),
        batch_size=args.batch_size, 
        num_workers=8)

num_classes = train_set.num_classes

# modify exist resnet101 model
model = models.resnet101(pretrained=False, num_classes=num_classes)
model.cuda()

#train setting 
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
criterion = nn.MSELoss()
eval_per_iter = 10
display_per_iter = 10
save_per_epoch = 5
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
        loss_li.append(loss.item())

        if i % eval_per_iter == eval_per_iter-1:
            pred_li = []
            for j, data_ in enumerate(test_loader, start=0):
                inputs_, labels_ = (d.to('cuda') for d in data_)
                predicted = model(inputs_)
                pred = precision(predicted, labels_)
                pred_li.append(pred.item())
            tqdm_iter.set_description('{} iter: Training loss={:.5f} precision={:.5f}'.format(i, np.mean(loss_li), np.mean(pred_li)))

    if epoch % save_per_epoch == 0:
        out_path = os.path.join(args.save_path, 'resnet101_'+str(epoch)+'.pt')
        os.makedirs(args.save_path, exist_ok=True) 
        torch.save(model, out_path)

print('Done: training')
