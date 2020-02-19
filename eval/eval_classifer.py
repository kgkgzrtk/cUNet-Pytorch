import argparse
import pickle
import os


import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange, tqdm

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int)
parser.add_argument('--image_root', type=str, default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--pkl_path', type=str, default='/mnt/data2/matsuzaki/repo/data/sepalated_data.pkl')
parser.add_argument('--output_dir', type=str, default='/mnt/fs2/2018/matsuzaki/results/eval/classifer_val')
parser.add_argument('--classifer_path', type=str, default='cp/classifier/i2w_res101_val_n/resnet101_95.pt')
#parser.add_argument('--classifer_path', type=str, default='cp/classifier/res_aug_5_cls/resnet101_95.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=5)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset

from dataset import ClassImageLoader
from sampler import ImbalancedDatasetSampler
from cunet import Conditional_UNet

if __name__=='__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    sep_data = pd.read_pickle(args.pkl_path)
    sep_data = sep_data['test']
    print('loaded {} data'.format(len(sep_data)))

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ClassImageLoader(paths=sep_data, transform=transform)
    
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.num_workers)

    #load model
    classifer= torch.load(args.classifer_path)
    classifer.eval()

    if args.gpu > 0:
        classifer.cuda()

    bs = args.batch_size
    s_li = ['sunny','cloudy', 'rain', 'snow', 'foggy'] 
    
    res_li = []
    for i, data in tqdm(enumerate(loader)):
        batch = data[0].to('cuda')
        c_batch = data[1].to('cuda')
        pred = torch.argmax(classifer(batch).detach(), 1)
        res_li.append(torch.cat([pred.int().cpu().view(bs,-1), c_batch.int().cpu().view(bs,-1)], 1))
    all_res = torch.cat(res_li, 0).numpy()
    y_true, y_pred = (all_res[:, 1], all_res[:, 0])

    table = classification_report(y_true, y_pred)

    print(table)

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(s_li)))
    
    df = pd.DataFrame(data=matrix, index=s_li, columns=s_li)
    df.to_pickle(os.path.join(args.output_dir,'cm.pkl'))

    plot = sns.heatmap(df, square=True, annot=True, fmt='d')

    fig = plot.get_figure()
    fig.savefig(os.path.join(args.output_dir,'pr_table.png'))
