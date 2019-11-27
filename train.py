import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str)
parser.add_argument('--name', type=str, default='cUNet')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_dir', type=str, default='cp')
parser.add_argument('--out_dir', type=str, default='results')
parser.add_argument('--pkl_path', type=str)
parser.add_argument('--estimator_path', type=str)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lmda', type=float, default=None)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
from utils import ImageLoader, FlickrDataLoader, _collate_fn
from cunet import Conditional_UNet
from disc import SNDisc
from sampler import ImbalancedDatasetSampler


class WeatherTransfer(object):

    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size
        self.global_step = 0

        os.makedirs(os.path.join(args.out_dir, args.name), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)
        comment = '_lr-{}_bs-{}_ne-{}_name-{}'.format(args.lr, args.batch_size, args.num_epoch, args.name)
        self.writer = SummaryWriter(comment=comment)

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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.input_size,)*2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform = {'train': train_transform, 'test': test_transform}
        self.train_set, self.test_set = self.load_data(varbose=True)
        self.num_classes = self.train_set.num_classes
        self.build()
        
    def load_data(self, varbose=False):

        print('Start loading a pickle file...')
        df = pd.read_pickle(args.pkl_path)
        print('loaded {} data'.format(len(df)))

        train_data_rate = 0.9
        pivot = int(len(df) * train_data_rate)
        df_shuffle = df.sample(frac=1)
        df_sep = {'train': df_shuffle[:pivot], 'test': df_shuffle[pivot:]}
        del df, df_shuffle
        cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
        loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], cols, transform=self.transform[s])
        train_set = loader('train')
        test_set = loader('test')

        print('train:{} test:{} sets have already loaded.'.format(len(train_set), len(test_set)))

        return train_set, test_set

    def build(self):
        args = self.args
        
        # Models
        self.inference = Conditional_UNet(num_classes=self.num_classes)
        self.discriminator = SNDisc(num_classes=self.num_classes)
        self.estimator = torch.load(args.estimator_path)
        self.estimator.eval()

        #Models to CUDA
        [i.cuda() for i in [self.inference, self.discriminator, self.estimator]]

        # Optimizer
        self.g_opt = torch.optim.Adam(self.inference.parameters(), lr=args.lr, betas=(0.0, 0.9), weight_decay=args.lr/50)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.0, 0.9), weight_decay=args.lr/50)

        self.train_loader = torch.utils.data.DataLoader(
                self.train_set,
                sampler=ImbalancedDatasetSampler(self.train_set),
                batch_size=args.batch_size,
                drop_last=True,
                num_workers=args.num_workers)

        self.test_loader = torch.utils.data.DataLoader(
                self.test_set,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=args.num_workers)

        self.random_loader = torch.utils.data.DataLoader(
                self.train_set,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers)

        self.seq_labels = get_sequential_labels(self.num_classes, args.batch_size)
        self.scalar_dict = {}
        self.image_dict = {}
        self.shift_lmda = lambda a,b: (1.-self.lmda)*a+self.lmda*b


    def update_inference(self, images, labels):
        #--- UPDATE(Inference) ---#
        self.g_opt.zero_grad()

        #for real
        pred_labels = self.estimator(images)
        real_res = self.discriminator(images, pred_labels.detach())
        real_d_out = real_res[0]
        real_feat = real_res[3]

        fake_out = self.inference(images, labels)
        fake_c_out = self.estimator(fake_out)
        fake_res = self.discriminator(fake_out, labels)
        fake_d_out = fake_res[0]
        fake_feat = fake_res[3]


        # Calc Generator Loss
        g_loss_adv = gen_hinge(fake_d_out)       # Adversarial loss
        g_loss_l1 = l1_loss(fake_out, images)
        g_loss_w = pred_loss(fake_c_out, labels)   # Weather prediction
        
        abs_loss = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        lmda, _ = torch.max(torch.abs(pred_labels.detach() - labels), 1)
        loss_con = torch.mean(abs_loss/(lmda+1e-7)) # Reconstraction loss
        
        g_loss = g_loss_adv + loss_con + g_loss_w
        
        g_loss.backward()
        self.g_opt.step()
        
        self.scalar_dict.update({
            'losses/g_loss/train': g_loss.item(),
            'losses/g_loss_adv/train': g_loss_adv.item(),
            'losses/g_loss_l1/train': g_loss_l1.item(),
            'losses/g_loss_w/train': g_loss_w.item(),
            'variables/lmda': self.lmda
            }) 

        self.image_dict.update({
            'io/train': torch.cat([images, fake_out], dim=3),
            })

    def update_discriminator(self, images, labels):

        #--- UPDATE(Discriminator) ---#
        self.d_opt.zero_grad()

        #for real
        real_c_out = self.estimator(images)
        pred_labels = real_c_out.detach()

        #real_d_out_same = self.discriminator(images, labels)
        real_d_out_pred = self.discriminator(images, pred_labels)[0]
        #real_d_out = self.shift_lmda(real_d_out_same, real_d_out_pred)

        #for fake
        fake_out = self.inference(images, labels)
        fake_d_out = self.discriminator(fake_out.detach(), labels)[0]
        
        #update
        d_loss = dis_hinge(fake_d_out, real_d_out_pred)
        d_loss.backward()
        self.d_opt.step()
        
        self.scalar_dict.update({
            'losses/d_loss': d_loss.item()
            })

    def evaluation(self):
        g_loss_ = []
        g_loss_adv_ = []
        g_loss_l1_ = []
        g_loss_w_ = []
        input_li = []
        fake_out_li = []
        data_iter = iter(self.test_loader)
        data_ = next(data_iter)
        images = data_[0].to('cuda')
        blank = torch.zeros_like(images[0]).unsqueeze(0)
        data_ = next(data_iter)
        ref, cond = (d.to('cuda') for d in data_)
        for i in range(self.batch_size):
            labels = torch.cat([cond[i]]*self.batch_size).view(-1, self.num_classes)
            with torch.no_grad():
                fake_out_ = self.inference(images, labels)
                fake_c_out_ = self.estimator(fake_out_)
                fake_d_out_ = self.discriminator(fake_out_, labels)[0]

            fake_out_li.append(fake_out_)
            g_loss_adv_.append(adv_loss(fake_d_out_, self.real).item())
            g_loss_l1_.append(l1_loss(fake_out_, images).item())
            g_loss_w_.append(pred_loss(fake_c_out_, labels).item())

        #--- WRITING SUMMARY ---#
        self.scalar_dict.update({
                'losses/g_loss_adv/test': np.mean(g_loss_adv_),
                'losses/g_loss_l1/test': np.mean(g_loss_l1_),
                'losses/g_loss_w/test': np.mean(g_loss_w_),
                }) 
        ref_img = torch.cat([blank]+list(torch.split(ref, 1)), dim=3)
        in_out_img = torch.cat( [images]+fake_out_li, dim=3)
        res_img = torch.cat([ref_img, in_out_img], dim=0)
        

        self.image_dict.update({
            'images/test': res_img,
                })

    def update_summary(self):
        # Summarize
        for k, v in self.scalar_dict.items():
            self.writer.add_scalar(k, v, self.global_step)
        for k, v in self.image_dict.items():
            grid = make_grid(v,
                    nrow=1,
                    normalize=True, scale_each=True)
            self.writer.add_image(k, grid, self.global_step)

    def train(self):
        args = self.args

        #train setting 
        eval_per_step = 100
        display_per_step = 100
        save_per_epoch = 5

        self.all_step = args.num_epoch*len(self.train_set)//self.batch_size

        tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
        for epoch in tqdm_iter:
            self.epoch = epoch
            for i, (data, rand_data) in enumerate(zip(self.train_loader, self.random_loader)):

                self.global_step += 1

                tqdm_iter.set_description('Training [ {} step ]'.format(self.global_step))
                if args.lmda:
                    self.lmda = args.lmda
                else:
                    self.lmda = self.global_step/self.all_step

                # Inputs
                images, _ = (d.to('cuda') for d in data)
                rand_images = rand_data[0].to('cuda')
                rand_labels = self.estimator(rand_images).detach()
                if images.size(0)!=self.batch_size: continue

                self.update_discriminator(images, rand_labels)
                self.update_inference(images, rand_labels)

                #--- EVALUATION ---#
                if (self.global_step % eval_per_step == 0):
                    self.evaluation()

                #--- UPDATE SUMMARY ---#
                if self.global_step % display_per_step == 0:
                    self.update_summary()

            if epoch % save_per_epoch == 0:
                out_path = os.path.join(args.save_dir, args.name, (args.name+'_e{:04d}.pt').format(epoch))
                state_dict = {
                        'inference': self.inference.state_dict(),
                        'discriminator': self.discriminator.state_dict()
                        }
                torch.save(state_dict, out_path)
        print('Done: training')

if __name__=='__main__':
    wt = WeatherTransfer(args)
    wt.train()
