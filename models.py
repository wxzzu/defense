import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from resnet import resnet18, resnet152
from wideresnet import WideResNet28, WideResNet34
from resnet50 import ResNet50
import torch.nn.init as init

class Generator(nn.Module):
    """ LeNets++ architecture from: "A Discriminative Feature Learning Approach for Deep Face Recognition"
        The variant is used by PCL, i.e. no max pooling and no padding.
    """
    def __init__(self, D):
        super(Generator, self).__init__()
        self.conv1 = self._make_conv_layer(1, 32, 5)
        self.conv2 = self._make_conv_layer(32, 32, 5)
        self.conv3 = self._make_conv_layer(32, 64, 5)
        self.conv4 = self._make_conv_layer(64, 64, 5)
        self.conv5 = self._make_conv_layer(64, 128, 5)
        self.conv6 = self._make_conv_layer(128, 128, 5)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size), nn.PReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.flatten(x, start_dim=1)
        return x


class GeneratorResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = resnet18()

    def forward(self, x):
        x = self.rn(x)
        return x

class GeneratorWideResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn =WideResNet34()

    def forward(self, x):
        x = self.rn(x)
        return x

class GeneratorResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn=ResNet50()


    def forward(self, x):
        x = self.rn(x)
        return x


class VanillaBase(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


class VanillaCNN(VanillaBase):
    def __init__(self, D, C):
        super().__init__()
        self.gen = Generator(D)
        self.fc1 = nn.Linear(2048, D)
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = self.fc1(x)
        x = self.proto(x)
        return x


class VanillaResNet18(VanillaBase):
    def __init__(self, D, C):
        super().__init__()
        # self.gen = GeneratorResNet18()
        # self.fc1 = nn.Linear(512, D)
        self.gen=GeneratorWideResNet34()
        self.fc1=nn.Linear(640,D)
#        self.gen = GeneratorResNet50()
#        self.fc1 = nn.Linear(2048, D)
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        x = self.proto(x)
        return x


class ResNet18_StochasticBaseDiagonal(nn.Module):
    """ Zero mean, trainable variance. """
    def __init__(self, D, disable_noise=False):
        super().__init__()
        # self.gen=GeneratorResNet50()
        # self.gen=GeneratorWideResNet34()
        self.gen = GeneratorResNet18()
        self.fc1=nn.Linear(512,D)
        # self.fc1=nn.Linear(640,D)
        # self.fc1=nn.Linear(2048,D)
        self.sigma = nn.Parameter(torch.rand(D), requires_grad=True)
        self.disable_noise = disable_noise

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        if not self.disable_noise:
            dist=Normal(0,f.softplus(self.sigma))
            x_sample = dist.rsample()
            #x =15*f.normalize(x, p=2, dim=1) 
            # * self.radius1
            x = x + x_sample
            self.radius1 = x.norm(p=2, dim=1, keepdim=True)  
            

            x =f.normalize(x, p=2, dim=1)* self.radius1
        return x


class ResNet18_StochasticBaseMultivariate(nn.Module):
    """ Trainable lower triangular matrix L, so Sigma=LL^T. """
    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen=GeneratorResNet50()
        # self.gen=GeneratorWideResNet34()
        # self.gen = GeneratorResNet18()
        # self.fc1 = nn.Linear(512, D)
        # self.fc1=nn.Linear(640,D)
        self.fc1=nn.Linear(2048,D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D).tril(), requires_grad=(not disable_noise))
        self.disable_noise = disable_noise

    @property
    def sigma(self):
        return self.L @ self.L.T

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        if not self.disable_noise:
            dist = MultivariateNormal(self.mu, scale_tril=self.L)
            x_sample = dist.rsample()
            x = x + x_sample
        return x

class WCANet_Base(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt",map_location="cuda:0"))

    def freeze_model_params(self):
        """ Freezes all model parameters except for the noise layer (used for ablation). """
        for param in self.base.gen.parameters():
            param.requires_grad = False
        for param in self.base.fc1.parameters():
            param.requires_grad = False
        for param in self.proto.parameters():
            param.requires_grad = False

    def unfreeze_model_params(self):
        """ Reverses `freeze_model_params`. """
        for param in self.base.gen.parameters():
            param.requires_grad = True
        for param in self.base.fc1.parameters():
            param.requires_grad = True
        for param in self.proto.parameters():
            param.requires_grad = True


class WCANet_CNN(WCANet_Base):
    def __init__(self, D, C, variance_type, disable_noise=False):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = StochasticBaseDiagonal(D, disable_noise=disable_noise)
        elif variance_type == 'anisotropic':
            self.base = StochasticBaseDiagonal(D, disable_noise=disable_noise)
            # self.base = StochasticBaseMultivariate(D, disable_noise=disable_noise)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x


class WCANet_ResNet18(WCANet_Base):
    def __init__(self, D, C, variance_type, disable_noise=False):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = ResNet18_StochasticBaseDiagonal(D, disable_noise=disable_noise)
        elif variance_type == 'anisotropic':
            self.base = ResNet18_StochasticBaseMultivariate(D, disable_noise=disable_noise)
        # self.proto=nn.Linear(H+NH,C)
        # self.proto = nn.Linear(D, C)
        self.weight = nn.Parameter(torch.empty(C, D))
        init.xavier_normal_(self.weight)
        self.b2 =nn.Parameter(torch.zeros(C))
        self.sigma2 = nn.Parameter(torch.rand(C, D), requires_grad=True)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        # x = f.relu(self.fc1(x))
        # if not self.disable_noise:
        std = torch.clamp(f.softplus(self.sigma2), min=1e-4, max=1.0)
        dist2=Normal(0,std)
        dist2_sample = dist2.rsample()
        noise_weight=self.weight+0.001*dist2_sample
                     # +0.001*dist2_sample
        # print('a',self.weight)
        # print('b',dist2_sample)
        self.radius2 = noise_weight.norm(p=2, dim=1, keepdim=True)  
        fc2_w = f.normalize(noise_weight, p=2, dim=1) * self.radius2

        #fc2_w = f.normalize(noise_weight, p=2, dim=1)
        classifier = [fc2_w, self.b2]
        x = f.linear(x, classifier[0], classifier[1])
        # x = x + x_sample
        # x = self.proto(x)
        return x

def model_factory(dataset, training_type, variance_type, feature_dim, num_classes):
    if variance_type is not None and variance_type not in ('isotropic', 'anisotropic'):
        raise NotImplementedError('Only "isotropic" and "anisotropic" variance types supported.')

    if dataset == 'cifar10':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = WCANet_ResNet18(feature_dim, num_classes, variance_type)
    elif dataset == 'cifar100':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = WCANet_ResNet18(feature_dim, num_classes, variance_type)
    elif dataset == 'svhn':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = WCANet_ResNet18(feature_dim, num_classes, variance_type)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
