
import torch.nn as nn

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
                               padding_mode=self.padding_mode).cuda()  # Co zrobic z paddingiem i stridem?
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
    def __init__(self, input_layer, h_channels, number_of_rsacm):
        super().__init__()
        self.h_channels = h_channels
        self.number_of_rsacm = number_of_rsacm
        self.rsacm_list = nn.ModuleList(
            [(RsacmBlock(input_layer, self.h_channels)) for i in range(0, self.number_of_rsacm)])

    def forward(self, x):
        self.x_list = []
        x_copy = x.clone()
        x = self.rsacm_list[0](x_copy)
        self.x_list.append(x)
        for i in range(1, self.number_of_rsacm):
            self.x_list.append(self.rsacm_list[i](x_copy))
            x = self.rsacm_list[i](x_copy) + x
        return x

    def override_forward(self, number_of_rsacm_to_use):
        self.number_of_rsacm_to_use = number_of_rsacm_to_use
        self.old_forward = self.forward
        self.forward = self.forward_partial

    def reset_forward(self):
        self.forward = self.old_forward

    def forward_partial(self, x):
        x_copy = x.clone()
        x = self.rsacm_list[0](x_copy)
        for i in range(1, self.number_of_rsacm_to_use):
            x = self.rsacm_list[i](x_copy) + x
        return x