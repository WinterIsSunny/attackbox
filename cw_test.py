#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:32:48 2018

"""


from CW import CW
from FGSM import FGSM
from OPT_attack import OPT_attack
from ZOO import ZOO
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data

net = MNIST()
net.cuda()
net = torch.nn.DataParallel(net, device_ids=[0])
load_model(net,'mnist_gpu.pt')
net.eval()
model = net.module if torch.cuda.is_available() else net
amodel = PytorchModel(model, bounds=[0,1], num_classes=10)
attack = CW(amodel)


train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
for i, (xi,yi) in enumerate(test_loader):
    xi_v=torch.autograd.Variable(xi)
    res= amodel.predict(xi_v)
    print(res.size())
    attack(xi,yi)
    