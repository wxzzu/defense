import argparse
import time
import numpy as np
import data
import models
import os
# import utils1
from datetime import datetime
import json
import os
import sys
import torchvision
import torch
import torchvision.transforms as transforms
#from attacks.one_pixel import one_pixel_attack
from data_loaders import get_data_loader
from models import model_factory
from test import test_attack
from train import train_vanilla, train_stochastic, train_stochastic_adversarial
from utils import attack_to_dataset_config
import torch

from foolbox import PyTorchModel
from foolbox.attacks import FGSMMC, PGDMC, BIMMC, CWMC
# import torchattacks
#


from foolbox import PyTorchModel
from foolbox.attacks import FGSMMC, PGDMC, BIMMC, CWMC
# import torchattacks
#
# from autoattack import AutoAttack
# model=model_factory("cifar10", "vanilla",None, 32, 10)
# model2=model_factory("cifar100", "vanilla",None, 256, 100)
model=model_factory("cifar10", "stochastic","anisotropic", 32, 10)
model3=model_factory("cifar100", "stochastic","anisotropic", 256, 100)
# model.load(os.path.join('C:/Users/dell/Desktop/WCA-Net-main_tsne/output/models/wcanet_cifar10_m2/0_01sigma', 'ckpt_last'))
# model.load(os.path.join(r'C:\Users\dell\Desktop\WCA-Net-main_tsne\output\stats\wcanet_cifar10_m2', 'ckpt_last'))
model.load(os.path.join('./output/models/wcanet_cifar10_m3/ckpt_best'))
model.eval()
model.cuda()
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]
epsilons=[1/255,0]
tranfrom_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
test_set = torchvision.datasets.CIFAR10(root=r'./data', train=False, download=False, transform=tranfrom_test)
preprocessing = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), axis=-3)
fbox_model = PyTorchModel(model, bounds=(0, 1), device='cuda', preprocessing=preprocessing)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True)
# test_loader = get_data_loader("cifar10",100, False, shuffle=False, drop_last=False)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as f
mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)
mean_cifar100 = (0.5071, 0.4867, 0.4408)
std_cifar100 = (0.2675, 0.2565, 0.2761)
def normalize_cifar10(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar10[0]) / std_cifar10[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar10[1]) / std_cifar10[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar10[2]) / std_cifar10[2]
    return t
def normalize_cifar100(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar100[0]) / std_cifar100[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar100[1]) / std_cifar100[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar100[2]) / std_cifar100[2]
    return t



for i,(data,target) in enumerate(test_loader):
    data,target=data.cuda(),target.cuda()
    data = normalize_cifar10(data)
    logits=model(data)
    # feature = model.base(data)
    # feature=model.sigma
    # feature=f.relu(model.fc1(model.gen(data)))
    break

import numpy as np
# np.random.seed(0)
# import seaborn as sns
# sns.set_theme()
# ax=sns.heatmap(feature.cpu().detach().numpy())
# ax=sns.heatmap(feature.cpu().detach().numpy(),vmin=0,vmax=1)
# ax=sns.heatmap(feature.cpu().detach().numpy(),annot=True)
# ax=sns.heatmap(feature.cpu().detach().numpy(),linewidth=.5)

# plt.savefig('sigma001.pdf')
# for i,(data,target) in enumerate(test_loader):
#     data,target=data.cuda(),target.cuda()
#     data = normalize_cifar10(data)
#
#     # feature=f.relu(model.fc1(model.gen(data)))
#     break
colors=['r','y','g','c','b','m','gray','peru','orange','pink']
tsne=TSNE(n_components=2)
Y=tsne.fit_transform(logits.cpu().detach().numpy())
a=[[],[],[],[],[],[],[],[],[],[]]
for i in range(10):
    for digit in Y[target.cpu()==i]:
        a[i].append(digit)
for i in range(10):
    plt.scatter(*np.array(a[i]).T, marker='.',color=colors[i],label='{}'.format(i))
plt.legend(loc = 'upper right')
plt.scatter(Y[:,0],Y[:,1],c=target.cpu(),cmap=plt.cm.Spectral)
plt.savefig('adist9.pdf')
plt.show()
print(logits.shape)




