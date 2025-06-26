import torch
from torch.utils.data import DataLoader
from cifar.resnet import ResNet32
from cifar.data_utils import *
import numpy as np
import scipy.io

# 设置参数
batch_size = 100
checkpoint_path = '//home/zy/pycharm/project/MetaSAug-main/ImageNet_iNat/models/resnet50_uniform_e90.pth.tar'  # 模型的路径，你需要替换成你保存的模型的路径
output_file = '/home/zy/pycharm/project/MetaSAug-main/yangzy/TEST_159_ckpte90.pth.mat'  # 特征保存的文件名，可以是.mat格式

# 建立数据加载器
_, train_dataset, test_dataset = build_dataset('cifar100', 100)  # 假设你使用的是CIFAR-100数据集

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 建立模型
model = ResNet32(num_classes=100)  # 假设你使用的是CIFAR-100数据集
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 提取特征
features_list = []
labels_list = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        # 使用模型前向传播获取特征
        features, _ = model(inputs, 4096)
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())

# 将特征保存为.mat文件
features_array = np.concatenate(features_list, axis=0)
labels_array = np.concatenate(labels_list, axis=0)
output_dict = {'features': features_array, 'labels': labels_array}
scipy.io.savemat(output_file, output_dict)

print("Features saved to", output_file)
