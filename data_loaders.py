import torch
from torchvision import datasets, transforms


def get_data_loader(dataset, batch_size, train=True,shuffle=True, drop_last=True):
    # Note that we do not normalize in the data loader, because we may use adv. examples
    # during training or testing.
    if dataset not in ('mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'):
        raise NotImplementedError('Dataset not supported.')
    if dataset == 'mnist':
        tr = transforms.Compose([
            transforms.ToTensor(),
        ])
        d = datasets.MNIST('./data', train=train,download=True,transform=tr)
    if dataset == 'fmnist':
        tr = transforms.Compose([
            transforms.ToTensor(),
        ])
        d = datasets.FashionMNIST('./data', train=train,download=True, transform=tr)
    elif dataset == 'cifar10':
        if train:
            tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]),
            ])
        d = datasets.CIFAR10('./data', train=train,download=True, transform=tr)
    elif dataset == 'cifar100':
        if train:
            tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        d = datasets.CIFAR100('./data', train=train,download=True, transform=tr)
    elif dataset == 'svhn':
        if train:
            tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        split = 'train' if train else 'test'
        d = datasets.SVHN('./data', split=split,download=True,transform=tr)
    data_loader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    #subset = torch.utils.data.Subset(full_dataset, range(1000))
    return data_loader
