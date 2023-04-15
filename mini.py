#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import wandb


# In[2]:


"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class _ModifiedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(_ModifiedResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(1024 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
    
def ModifiedResNet():
    return ResNet(BasicBlock, [1, 1, 1, 1])

def ModifiedResNet_1():
    return _ModifiedResNet(BasicBlock, [1, 1, 1])

def ModifiedResNet_2():
    return _ModifiedResNet(BasicBlock, [2, 2, 2])


# In[3]:


model = ResNet18().cuda()
summary(model, (3, 32, 32))


# In[4]:


model = ModifiedResNet().cuda()
summary(model, (3, 32, 32))


# In[5]:


model = ModifiedResNet_1().cuda()
summary(model, (3, 32, 32))


# In[6]:


model = ModifiedResNet_2().cuda()
summary(model, (3, 32, 32))


# In[7]:


def create_data_loaders(data_path, num_workers):
    # reference: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L30
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=num_workers
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def create_optimizer(model, optimizer_name, lr=0.1, momentum=0.9, weight_decay=5e-4):
    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd-nesterov":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adadelta":
        return optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    return None


# training the model
def train(model, train_loader, run_train, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if run_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            num_batches = batch_idx

    train_loss = train_loss / (num_batches + 1)
    train_accuracy = 100.0 * (correct / total)

    return train_loss, train_accuracy


def test(model, test_loader, criterion, optimizer, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            num_batches = batch_idx

    test_loss = test_loss / (num_batches + 1)
    test_accuracy = 100.0 * (correct / total)

    return test_loss, test_accuracy


# In[8]:


def run(
    model,
    use_cuda,
    data_path,
    run_train,
    num_data_loader_workers,
    num_epochs,
    optimizer,
    lr,
    weight_decay=5e-4,
    momentum=0.9,
):
    # creating train/test data loaders
    train_loader, test_loader = create_data_loaders(data_path, num_data_loader_workers)

    # check if cuda is available and can be used
    # configure model to use cuda device accordingly
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer, lr, momentum, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(
            model, train_loader, run_train, criterion, optimizer, device
        )
        test_loss, test_accuracy = test(
            model, test_loader, criterion, optimizer, device
        )
        print(
            f"Epoch: {epoch + 1} | [Train] Loss: {train_loss:.2f}, Acccuracy: {train_accuracy:.2f} | [Test] Loss: {test_loss:.2f}, Acccuracy: {test_accuracy:.2f}"
        )
        if wandb.run is not None:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "epoch": epoch + 1,
                }
            )
        if run_train:
            scheduler.step()
    return model


# In[9]:


def tune_parameters(config=None):
    with wandb.init(config=config):
        config = wandb.config
        model = ModifiedResNet()
        run(
            use_cuda=True,
            data_path="data/",
            run_train=True,
            num_data_loader_workers=4,
            num_epochs=25,
            optimizer=config.optimizer,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )


# In[10]:


metric_config = {
    "name": "test_accuracy",
    "goal": "maximize",
}

parameters_config = {
    "optimizer": {
        "values": ["adam", "sgd", "sgd-nesterov", "adadelta", "adagrad"],
    },
    "learning_rate": {"values": [0.001, 0.005, 0.01, 0.05, 0.1]},
    "weight_decay": {"values": [0.0001, 0.0005, 0.001, 0.005]},
    "momentum": {"values": [0.9, 0.99]},
}

sweep_config = {
    "method": "random",
    "metric": metric_config,
    "parameters": parameters_config,
}

import pprint

pprint.pprint(sweep_config)


# In[11]:


# wandb.login()
# sweep_id = wandb.sweep(sweep_config, project="resnet-cifar")
# wandb.agent(sweep_id, tune_parameters, entity="aincrad", project="resnet-cifar", count=50)


# In[ ]:


wandb.login()


# In[14]:


wandb.init(name="ModifiedResNet", project="resnet-cifar", entity="rgg9776")
model = run(
    model=ModifiedResNet(),
    use_cuda=True,
    data_path="data/",
    run_train=True,
    num_data_loader_workers=8,
    num_epochs=200,
    optimizer="adagrad",
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
)


# In[ ]:


wandb.init(name="ModifiedResNet-1", project="resnet-cifar", entity="rgg9776")
model = run(
    model=ModifiedResNet_1(),
    use_cuda=True,
    data_path="data/",
    run_train=True,
    num_data_loader_workers=8,
    num_epochs=200,
    optimizer="adagrad",
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
)


# In[ ]:


wandb.init(name="ModifiedResNet-2", project="resnet-cifar", entity="rgg9776")
model = run(
    model=ModifiedResNet_2(),
    use_cuda=True,
    data_path="data/",
    run_train=True,
    num_data_loader_workers=8,
    num_epochs=200,
    optimizer="adagrad",
    lr=0.1,
    weight_decay=5e-4,
    momentum=0.9,
)

