
import torch.nn as nn
import torch
from models.gating_network import Gating_Network
from ptflops import get_model_complexity_info

class RsacmBlock(nn.Module):
    def __init__(self, input_layer: nn.Conv2d, h_channels: int):
        super().__init__()
        self.h_channels = h_channels
        self.in_channels = input_layer.in_channels
        self.out_channels = input_layer.out_channels
        self.kernel_size = input_layer.kernel_size
        self.stride = input_layer.stride
        self.padding = input_layer.padding
        self.groups = input_layer.groups
        self.bias = input_layer.bias
        self.padding_mode = input_layer.padding_mode
        self.dilation = input_layer.dilation
        self.conv1 = nn.Conv2d(self.in_channels, self.h_channels, 1, stride=1, padding=0, dilation=self.dilation,
                               groups=self.groups, bias=self.bias,
                               padding_mode=self.padding_mode).cuda()
        self.conv2 = nn.Conv2d(self.h_channels, self.out_channels, self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation,
                               # groups=self.groups if self.groups > 1 else self.h_channels,
                               groups=self.groups,
                               bias=self.bias, padding_mode=self.padding_mode).cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AccmBlock(nn.Module):
    def __init__(self, input_layer, h_channels, number_of_rsacm, args, rsacm_cost):
        super().__init__()
        self.h_channels = h_channels
        self.number_of_rsacm = number_of_rsacm
        self.args = args
        self.rsacm_list = nn.ModuleList(
            [(RsacmBlock(input_layer, self.h_channels)) for i in range(0, self.number_of_rsacm)])
        self.rsacm_cost = rsacm_cost

    def initialize_gating_network(self, activation, args):
        self.gating_network =  Gating_Network(activation, args)
        macs, _ = get_model_complexity_info(self.gating_network, (activation.shape[1], activation.shape[2], activation.shape[3]), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
        gflops = 2*macs
        self.gating_network_cost = gflops

    def forward(self, x):
        if self.args.train_gating_networks:
            gating_network_output =  self.gating_network(x)
            if self.args.training_type in ['eval_gating_networks', 'calculate_complexity']:
                self.gating_network_output_detached = gating_network_output.detach()
            costs = torch.tensor([self.rsacm_cost*i for i in range(1,self.number_of_rsacm+1)]).cuda()
        else:
            x = x.detach() # Detach x make sure that gradient for this layer won't flow to previous layers.
        self.x_list = []
        x_copy = x.clone()
        x = self.rsacm_list[0](x_copy)
        self.x_list.append(x)
        for i in range(1, self.number_of_rsacm):
            rsacm_output = self.rsacm_list[i](x_copy)
            self.x_list.append(rsacm_output + x.detach()) # Detach x before adding it to the sum
            x = rsacm_output + x
        if self.args.train_gating_networks:
            self.x_list_tensor = torch.stack(self.x_list)
            result = torch.einsum('nrchw,nr->nchw', self.x_list_tensor.permute(1,0,2,3,4), gating_network_output)
            self.loss_cost_gn = torch.sum((gating_network_output * costs)) / (gating_network_output.shape[0]*self.rsacm_cost*(self.number_of_rsacm))
            return result
        return x

    def override_forward(self, number_of_rsacm_to_use):
        self.number_of_rsacm_to_use = number_of_rsacm_to_use
        self.old_forward = self.forward
        self.forward = self.forward_partial

    def reset_forward(self):
        self.forward = self.old_forward

    def forward_partial(self, x):
        # Used only for evaluation!
        x_copy = x.clone()
        x = self.rsacm_list[0](x_copy)
        for i in range(1, self.number_of_rsacm_to_use):
            x = self.rsacm_list[i](x_copy) + x
        return x