import torchvision.models as models
from utils.models import get_module_by_name
from models.accm import AccmBlock
import torch
from copy import copy
import glob
from itertools import chain
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int = 10, extension: int = 1):
        super().__init__()
        self.input_channels = 3
        self._num_classes = num_classes
        self.in_planes = 64 * extension
        num_channels = 64 * extension
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.layer1 = self._make_layer(block, num_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * num_channels, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * num_channels, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * num_channels, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * num_channels * block.expansion, num_classes)
        self.activation = {}
        self.inputs = {}


    def get_activation(self, name):
            def hook(model, input, output):
                self.activation[name] = output
                self.inputs[name] = input

            return hook

    def register_all_hooks(self):
        for idx, value in enumerate(self.replaced_modules):
            get_module_by_name(self, value).register_forward_hook(self.get_activation(idx))

    def override_forward_for_accm_blocks(self, number_of_rsacm):
        for idx, value in enumerate(self.replaced_modules):
            get_module_by_name(self, value).override_forward(number_of_rsacm)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward_generator(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        for block in chain(self.layer1, self.layer2, self.layer3, self.layer4):
            x = block(x)
            x = yield x, None
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        _ = yield None, x

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def load_teacher_model(args):
    args_copy = copy(args)
    args_copy.accm = False
    
    teacher_model = ResNet50().cuda()
    model_file = glob.glob("./pre-trained/resnet_bartek" + "/*.pth")
    print(f'Using Pretrained model {model_file[0]} for the teacher model')
    checkpoint = torch.load(model_file[0])
    teacher_model.load_state_dict(checkpoint['model_state'])
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model

def ResNet18(num_classes: int = 10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes: int = 10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)