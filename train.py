import argparse
import pickle
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from ops import *
from utils import ImageLoader, _collate_fn
from cunet import Conditional_UNet
from disc import Discriminator
from sampler import ImbalancedDatasetSampler

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str, default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--name', type=str, default='cUNet')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_dir', type=str, default='cp')
parser.add_argument('--out_dir', type=str, default='results')
parser.add_argument('--pkl_path', type=str, default='data_pkl/sepalated_mini_data.pkl')
parser.add_argument('--classifier_path', type=str, default='cp/resnet101_5.pt')
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=1)

class WeatherTransfer(object):

    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size

        os.makedirs(os.path.join(args.out_dir, args.name), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)
        self.writer = SummaryWriter()

        # Consts
        self.real = Variable_Float(1., self.batch_size)
        self.fake = Variable_Float(0., self.batch_size)
        self.lmda = 0.

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
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.input_size,)*2),
            transforms.ToTensor()
        ])
        self.transform = {'train': train_transform, 'test': test_transform}
        self.train_set, self.test_set = self.load_data(varbose=True)
        self.num_classes = len(self.train_set.classes)
        self.build()
        
    def load_data(self, varbose=False):

        print('Start loading a pickel file...')
        with open(args.pkl_path, 'rb') as f:
            sep_data = pickle.load(f)

        loader = lambda s: ImageLoader(paths=sep_data[s], transform=self.transform[s])
        train_set = loader('train')
        test_set = loader('test')

        print('train:{} test:{} sets have already loaded.'.format(len(train_set), len(test_set)))

        return train_set, test_set

    def build(self):
        args = self.args
        
        # Models
        self.inference = Conditional_UNet(num_classes=self.num_classes)
        self.discriminator = Discriminator(num_classes=self.num_classes)
        self.classifier = torch.load(args.classifier_path)
        self.classifier.eval()

        #Models to CUDA
        [i.cuda() for i in [self.inference, self.discriminator, self.classifier]]

        # Optimizer
        self.g_opt = torch.optim.Adam(self.inference.parameters(), lr=args.lr, weight_decay=1e-4)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, weight_decay=1e-4)

        self.train_loader = torch.utils.data.DataLoader(
                self.train_set,
                sampler=ImbalancedDatasetSampler(self.train_set),
                batch_size=args.batch_size,
                drop_last=True,
                num_workers=args.num_workers)

        self.test_loader = torch.utils.data.DataLoader(
                self.test_set,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers)

        self.seq_labels = get_sequential_labels(self.num_classes, args.batch_size)
        self.scalar_dict = {}
        self.image_dict = {}


    def update_inference(self, images, labels):
        #--- UPDATE(Inference) ---#
        self.g_opt.zero_grad()
        fake_out = self.inference(images, labels)
        fake_c_out = self.classifier(fake_out)
        fake_d_out = self.discriminator(fake_out, labels)

        # Calc Generator Loss
        g_loss_adv = adv_loss(fake_d_out, self.real)       # Adversarial loss
        g_loss_l1 = l1_loss(fake_out, images) #TODO seg_loss
        g_loss_w = pred_loss(fake_c_out, labels)   # Weather prediction
        g_loss = g_loss_adv + (1 - self.lmda)*g_loss_l1 + self.lmda*g_loss_w
        g_loss.backward()
        self.g_opt.step()

        self.scalar_dict.update({
                'losses/g_loss/train':  g_loss.item(),
                'losses/g_loss_adv/train': g_loss_adv.item(),
                'losses/g_loss_l1/train': g_loss_l1.item(),
                'losses/g_loss_w/train': g_loss_w.item(),
                }) 

        self.image_dict.update({
                'images/train': images,
                'labels/train': labels,
                'outputs/train': fake_out,
                })

    def update_discriminator(self, images, labels):

        #--- UPDATE(Discriminator) ---#
        self.d_opt.zero_grad()
        #for real
        real_c_out = self.classifier(images)
        real_d_out = self.discriminator(images, real_c_out)
        d_real_loss = adv_loss(real_d_out, self.real)
        #for fake
        fake_out = self.inference(images, labels)
        fake_d_out = self.discriminator(fake_out.detach(), labels)
        d_fake_loss = adv_loss(fake_d_out, self.fake)
        #update
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_opt.step()
        
        self.scalar_dict.update({
            'losses/d_loss/train': d_loss.item(),
            'losses/d_loss_real':   d_real_loss.item(),
            'losses/d_loss_fake':   d_fake_loss.item()
            })

    def eval(self, k=10):
        #--- EVALUATION ---#
        g_loss_ = []
        g_loss_adv_ = []
        g_loss_l1_ = []
        g_loss_w_ = []
        input_li = []
        fake_out_li = []
        with torch.no_grad():
            for i, data_ in enumerate(self.test_loader):
                if i>=k: break
                images, labels = (d.to('cuda') for d in data_)
                fake_out_ = self.inference(images, self.seq_labels)
                fake_c_out_ = self.classifier(fake_out_)
                fake_d_out_ = self.discriminator(fake_out_, self.seq_labels)
                input_li.append(images)
                fake_out_li.append(fake_out_)
                g_loss_adv_.append(adv_loss(fake_d_out_, self.real).item())
                g_loss_l1_.append(l1_loss(fake_out_, images).item())
                g_loss_w_.append(pred_loss(fake_c_out_, self.seq_labels).item())

        #--- WRITING SUMMARY ---#
        self.scalar_dict.update({
                'losses/g_loss_adv/test': np.mean(g_loss_adv_),
                'losses/g_loss_l1/test': np.mean(g_loss_l1_),
                'losses/g_loss_w/test': np.mean(g_loss_w_),
                }) 

        self.image_dict.update({
                'inputs/test': input_li[0],
                'labels/test': self.seq_labels,
                'outputs/test': fake_out_li[0],
                })

    def update_summary(self):
        # Summarize
        for k, v in self.scalar_dict.items():
            self.writer.add_scalar(k, v, self.epoch)
        for k, v in self.image_dict.items():
            grid = make_grid(v, nrow=4,
                    normalize=True, scale_each=True)
            self.writer.add_image(k, grid, self.epoch)

    def train(self):

        #train setting 
        eval_per_epoch = 1
        display_per_iter = 10
        output_per_epoch = 1
        save_per_epoch = 5

        tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
        for epoch in range(args.num_epoch):
            self.epoch = epoch
            for i, data in tqdm(enumerate(self.train_loader)):

                self.global_step = (i+1)*(epoch*1)

                # Inputs
                images, _ = (d.to('cuda') for d in data)
                rand_labels = get_rand_labels(self.num_classes, self.batch_size)

                if images.size(0)!=self.batch_size: continue

                self.update_inference(images, rand_labels)
                self.update_discriminator(images, rand_labels)


                #--- UPDATE SUMMARY ---#
                if i % display_per_iter == 0:
                    self.update_summary()

            if epoch % eval_per_epoch == 0:
                self.eval()

            if epoch % save_per_epoch == 0:
                out_path = os.path.join(args.save_dir, args.name, (args.name+'_e{:04d}.pt').format(epoch))
                state_dict = {
                        'inference': self.inference.state_dict(),
                        'discriminator': self.discriminator.state_dict()
                        }
                torch.save(state_dict, out_path)
        print('Done: training')

if __name__=='__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    wt = WeatherTransfer(args)
    wt.train()
