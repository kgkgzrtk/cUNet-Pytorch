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
parser.add_argument('--estimator_path', type=str)
parser.add_argument('--input_size', type=int, default=244)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=6)
parser.add_argument('--num_frames', type=int, default=10)
parser.add_argument('--alpha', type=int, default=2)

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
    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageLoader(paths, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)

    #load model
    transfer = Conditional_UNet(num_classes=args.num_classes)
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])
    transfer.eval()

    estimator = torch.load(args.estimator_path)
    estimator.eval()

    if args.gpu > 0:
        transfer.cuda()
        estimator.cuda()

    bs = args.batch_size
    nf = args.num_frames
    eye = torch.eye(args.num_classes)

    for i, data in enumerate(loader):
        batch = data[0].to('cuda')
        tables = []
        scale = args.alpha
        for theta in np.arange(-np.pi/2, np.pi/2+np.pi/nf, np.pi/(nf-1)):
            scaled_one_hot = eye*torch.sin(torch.tensor(theta).float())*scale
            pred = estimator(batch)
            feats = [make_grid(batch, nrow=1, normalize=True, scale_each=True)]
            for axis, one_hot in enumerate(torch.split(scaled_one_hot, 1)):
                eye_ = torch.cat([1. - eye[axis]]*bs).to('cuda').view(-1, args.num_classes)
                c_batch = torch.cat([one_hot]*bs).to('cuda')
                c_batch += eye_*pred
                res = transfer(batch, c_batch).detach()
                res = (res + 1.)*127.5
                feats.append(make_grid(res, nrow=1, normalize=True, scale_each=True))
            tables.append(torch.cat(feats, 2))
        img_arr = [transforms.ToPILImage()(t.cpu()).convert("RGB") for t in tables]
        out_path = os.path.join(args.output_dir, 'output{}.gif'.format(i))
        print('Save gif image: {}'.format(out_path))
        img_arr[0].save(
                out_path,
                save_all=True,
                append_images=img_arr[1:]+img_arr[1:-1][::-1],
                duration=1000//nf,
                loop=0
                )
