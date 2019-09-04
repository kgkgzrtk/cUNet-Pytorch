import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
#import cupy as cp


xp = np # cpu or gpu
#if os.environ.get('CUDA_VISIBLE_DEVICES') is not None: xp = cp


def adv_loss(a, b):
    assert a.size() == b.size(), 'The size of a and b is different.{}!={}'.format(a.size(), b.size())
    return F.mse_loss(a, b)

def l1_loss(a, b):
    assert a.size() == b.size(), 'The size of a and b is different.{}!={}'.format(a.size(), b.size())
    return F.l1_loss(a, b)

def pred_loss(preds, labels):
    _, c = preds.size()
    max_ind = torch.argmax(preds, dim=1)
    one_hot_ = torch.zeros_like(preds)
    one_hot_[:, max_ind] = preds[:, max_ind]
    loss = adv_loss(one_hot_, labels)
    return loss

def get_rand_labels(num_classes, batch_size):
    rand_ind = torch.randint(num_classes, (batch_size,))
    one_hot = F.one_hot(rand_ind, num_classes).float().to('cuda')
    return one_hot

def get_sequential_labels(num_classes, batch_size):
    arr = xp.eye(num_classes, dtype=xp.float32)
    repeat = batch_size//num_classes + 1
    arr = xp.tile(arr, (repeat, 1))[:batch_size]
    return torch.from_numpy(arr).float().to('cuda')

def Variable_Float(x, batch_size):
    return Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(x), requires_grad=False)


