from turtle import forward
import torch.nn as nn
import torch


class Gating_Network(nn.Module):
    def __init__(self, activation, args):
        super().__init__()
        m = nn.Conv2d(activation.shape[1], 30, 1, stride=2).cuda()
        k = m(activation)
        self.neurons_fc = k.shape[1] * k.shape[2] * k.shape[3]
        self.first_conv = nn.Conv2d(activation.shape[1], 30, 1, stride=2).cuda()
        self.fc = nn.Linear(self.neurons_fc, args.number_of_rsacm).cuda()

    def forward(self, x):
        x = self.first_conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.nn.functional.gumbel_softmax(x, tau=1, hard=True)
        return x


def initialized_gating_networks(teacher_model, example_data, args):
    gating_networks = {}
    teacher_model(example_data)
    for index, module_name in enumerate(teacher_model.replaced_modules):
        gating_networks[module_name] = Gating_Network(teacher_model.inputs[index][0], args)
    return gating_networks