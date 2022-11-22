from turtle import forward
import torch.nn as nn
import torch


class Gating_Network(nn.Module):
    def __init__(self, activation, args):
        super().__init__()
        m = nn.Conv2d(activation.shape[1], 128, 1, stride=1).cuda()
        k = m(activation)
        self.neurons_fc = k.shape[1] * k.shape[2] * k.shape[3]
        self.first_conv = nn.Conv2d(activation.shape[1], 128, 1, stride=1).cuda()
        self.fc = nn.Linear(self.neurons_fc, args.number_of_rsacm).cuda()

    def forward(self, x):
        x = self.first_conv(x)
        # x = nn.ReLU()(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.nn.functional.gumbel_softmax(x, tau=1, hard=True)
        return x