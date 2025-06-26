import os
import numpy as np
import scipy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.manifold import TSNE
from attacks.fgsm import fgsm
from attacks.pgd import pgd
from metrics import accuracy
from utils import normalize_cifar10, normalize_cifar100, normalize_generic
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
class HELoss(nn.Module):
    def __init__(self, s=None):
        super(HELoss, self).__init__()
        self.s = 15

    def forward(self, logits, labels, cm):
        numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - cm)
        item = torch.cat([torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * item), dim=1)
        Loss = -torch.mean(numerator - torch.log(denominator))
        return Loss

def get_stochastic_model_optimizer(model, args):
    if args['var_type'] == 'isotropic':
        trainable_noise_params = {'params': [model.base.sigma,model.sigma2], 'lr': args['lr'], 'weight_decay': args['wd']}
    elif args['var_type'] == 'anisotropic':
        trainable_noise_params = {'params': model.sigma, 'lr': args['lr'], 'weight_decay': args['wd']}
        # trainable_noise_params = {'params': model.base.dist, 'lr': args['lr'], 'weight_decay': args['wd']}
    optimizer = Adam([
        {'params': model.base.gen.parameters(), 'lr': args['lr']},
        {'params': model.base.fc1.parameters(), 'lr': args['lr']},
        trainable_noise_params,
        {'params': [model.weight, model.b2], 'lr': args['lr'], 'weight_decay': args['wd']}
    ])
    return optimizer


def get_norm_func(args):
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'cifar100':
        norm_func = normalize_cifar100
    elif args['dataset'] == 'svhn':
        norm_func = normalize_generic
    elif args['dataset'] in ('mnist', 'fmnist'):
        norm_func = None
    return norm_func


def train_vanilla(model, train_loader, test_loader, args, device='cpu'):
    checkpoint_path = './output/models/wcanet_cifar10_m0/ckpt_best.pt'  # 模型的路径，你需要替换成你保存的模型的路径

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load("./output/models/wcanet_cifar10_m0/ckpt_best.pt", map_location=device))  # 替换为你自己的模型路径
    model.to(device)
    model.eval()

    # 提取特征
    features_list = []
    labels_list = []

    tranfrom_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = torchvision.datasets.CIFAR10(root=r'./data', train=False, download=False, transform=tranfrom_test)
    preprocessing = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), axis=-3)
    # fbox_model = PyTorchModel(model, bounds=(0, 1), device='cuda', preprocessing=preprocessing)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True)
    # test_loader = get_data_loader("cifar10",100, False, shuffle=False, drop_last=False)
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # import torch.nn.functional as f
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

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data = normalize_cifar10(data)
        logits = model(data)
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
    colors = ['r', 'y', 'g', 'c', 'b', 'm', 'gray', 'peru', 'orange', 'pink']
    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(logits.cpu().detach().numpy())
    a = [[], [], [], [], [], [], [], [], [], []]
    for i in range(10):
        for digit in Y[target.cpu() == i]:
            a[i].append(digit)
    for i in range(10):
        plt.scatter(*np.array(a[i]).T, marker='.', color=colors[i], label='{}'.format(i))
    plt.legend(loc='upper right')
    plt.scatter(Y[:, 0], Y[:, 1], c=target.cpu(), cmap=plt.cm.Spectral)
    plt.savefig('adist9.pdf')
    plt.show()
    print(logits.shape)


def train_stochastic(model, train_loader, test_loader, args, device='cpu'):
    optimizer = get_stochastic_model_optimizer(model, args)
    # optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = StepLR(optimizer, int(args['num_epochs'] / 3), 0.1)
    # Uncomment for the "train model and noise separately" ablation. But first train a model with disable_noise=True.
    # model.freeze_model_params()
    loss_func = nn.CrossEntropyLoss()
    norm_func = get_norm_func(args)
    best_test_acc = -1.
    for epoch in range(args['num_epochs']):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
            logits = model(data)
            optimizer.zero_grad()
            wca=model.sigma.sum()
            wca1=model.sigma2.sum()
            # ce_loss = HELoss(s=15)
# Compute the loss by passing logits, target (labels), and cm (optional) to the `forward()` method
            loss1 =  loss_func(logits, target)
            #ce_loss = HELoss(logits, target, 0.2)
            loss = loss1 -torch.log(wca)-torch.log(wca1)
            loss.backward()
            optimizer.step()
            # if args['var_type'] == 'anisotropic':
            #     with torch.no_grad():
            #         model.base.L.data = model.base.L.data.tril()
        scheduler.step()
        train_acc = accuracy(model, train_loader, device=device, norm=norm_func)
        test_acc = accuracy(model, test_loader, device=device, norm=norm_func)
        print('Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(os.path.join(args['output_path']['models'], 'ckpt_best'))
            print('Best test accuracy achieved on epoch {}.'.format(epoch + 1))
        model.save(os.path.join(args['output_path']['models'], 'ckpt_last'))


def train_stochastic_adversarial(model, train_loader, test_loader, args, device='cpu'):



    checkpoint_path = './output/models/wcanet_cifar10_m3/ckpt_best.pt'  # 模型的路径，你需要替换成你保存的模型的路径

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("./output/models/wcanet_cifar10_m3/ckpt_best.pt", map_location=device))  # 替换为你自己的模型路径
    model.to(device)
    model.eval()

    # 提取特征
    features_list = []
    labels_list = []

    tranfrom_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = torchvision.datasets.CIFAR10(root=r'./data', train=False, download=False, transform=tranfrom_test)
    preprocessing = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), axis=-3)
    #fbox_model = PyTorchModel(model, bounds=(0, 1), device='cuda', preprocessing=preprocessing)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True)
    # test_loader = get_data_loader("cifar10",100, False, shuffle=False, drop_last=False)
    #from sklearn.manifold import TSNE
   # import matplotlib.pyplot as plt
    #import torch.nn.functional as f
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

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data = normalize_cifar10(data)
        logits = model(data)
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
    colors = ['r', 'y', 'g', 'c', 'b', 'm', 'gray', 'peru', 'orange', 'pink']
    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(logits.cpu().detach().numpy())
    a = [[], [], [], [], [], [], [], [], [], []]
    for i in range(10):
        for digit in Y[target.cpu() == i]:
            a[i].append(digit)
    for i in range(10):
        plt.scatter(*np.array(a[i]).T, marker='.', color=colors[i], label='{}'.format(i))
    plt.legend(loc='upper right')
    plt.scatter(Y[:, 0], Y[:, 1], c=target.cpu(), cmap=plt.cm.Spectral)
    plt.savefig('adist9.pdf')
    plt.show()
    print(logits.shape)











