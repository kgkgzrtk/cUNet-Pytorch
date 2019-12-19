import argparse
import pickle
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int)
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--cp_path', type=str)
parser.add_argument('--input_size', type=int, default=244)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=6)
parser.add_argument('--num_frames', type=int, default=10)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from dataset import ImageLoader
from cunet import Conditional_UNet


if __name__=='__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    paths = [os.path.join(args.input_dir, fn)for fn in os.listdir(args.input_dir)]
    print(len(paths))
    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageLoader(paths, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)


    transfer = Conditional_UNet(num_classes=args.num_classes)
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])
    transfer.eval()
    if args.gpu > 0:
        transfer.cuda()
    
    bs = args.batch_size
    nf = args.num_frames
    for i, data in enumerate(loader):
        batch = data[0].to('cuda')
        tables = []
        for lmda in np.arange(0, 1+1/nf, 1/(nf-1)):
            eye = torch.eye(args.num_classes)*((lmda-0.5)*2)
            feats = []
            for one_hot in torch.split(eye, 1):
                c_batch = torch.cat([one_hot]*bs).to('cuda')
                res = transfer(batch, c_batch)
                feats.append(res)
            tables.append(torch.cat(feats, 3))
        img_arr = [transform.ToPILImage()(t) for t in tables]
        img_arr[0].save(
                os.path.join(args.output_dir, 'output{}.gif'.format(i)),
                save_all=True,
                append_images=img_arr[1:]
                )
