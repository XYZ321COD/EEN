import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):

    def __init__(self, base_model, out_dim, pretrained, args=None):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=pretrained, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=pretrained, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        self.args = args
        # self.backbone.conv1 = nn.Conv2d(3, 64, 3, stride=(2, 2), padding=(3, 3), bias=False)
    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        if self.args.save_feature_map:
            self.outputconv1 = self.backbone.conv1(x)
        x = self.backbone(x)
        return x

    def replace_with_accm(self):
        if self.args.accm:
            self.backbone.conv1 = AccmBlock(self.backbone.conv1, self.args.h_channels, self.args.number_of_rsacm)
        else:
            nn.init.xavier_uniform(self.backbone.conv1.weight.data)




class RsacmBlock(nn.Module):
    def __init__(self, input_layer: nn.Conv2d, h_channels):
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
        self.conv1 = nn.Conv2d(self.in_channels, self.h_channels, 1 , stride=1, padding=0, dilation=self.dilation, groups=self.groups, bias=self.bias, padding_mode=self.padding_mode).cuda() # Co zrobic z paddingiem i stridem?
        self.conv2 = nn.Conv2d(self.h_channels, self.out_channels, self.kernel_size , stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=self.bias, padding_mode=self.padding_mode).cuda()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AccmBlock(nn.Module):
    def __init__(self, input_layer, h_channels, number_of_rsacm):
        super().__init__()
        self.h_channels = h_channels
        self.number_of_rsacm = number_of_rsacm
        self.rsacm_list = nn.ModuleList([(RsacmBlock(input_layer, self.h_channels)) for i in range(0, self.number_of_rsacm)])

    def forward(self, x):
        x_copy = x.clone()
        x = self.rsacm_list[0](x_copy)
        for i in range(1, self.number_of_rsacm):
            x += self.rsacm_list[i](x_copy)
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
            x += self.rsacm_list[i](x_copy)
        return x
