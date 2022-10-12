import torch.nn as nn
import torchvision.models as models
from utils.models import get_module_by_name
from models.accm import AccmBlock
import torch
from copy import copy
import glob

class ResNet(nn.Module):

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
            self.inputs[name] = input

        return hook

    def __init__(self, base_model, out_dim, pretrained, args=None):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=pretrained, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=pretrained, num_classes=out_dim)}

        self.activation = {}
        self.inputs = {}

        self.backbone = self._get_basemodel(base_model)
        self.args = args

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.backbone(x)
        return x

    def register_all_hooks(self):
        for idx, value in enumerate(self.replaced_modules):
            get_module_by_name(self, value).register_forward_hook(self.get_activation(idx))

    def override_forward_for_accm_blocks(self, number_of_rsacm):
        for idx, value in enumerate(self.replaced_modules):
            get_module_by_name(self, value).override_forward(number_of_rsacm)

def load_teacher_model(args):
    args_copy = copy(args)
    args_copy.accm = False
    
    teacher_model = ResNet(base_model=args_copy.arch, out_dim=args_copy.num_classes, pretrained=args_copy.pretrained, args=args_copy).cuda()
    model_file = glob.glob("./pre-trained/resnet50_long" + "/*.tar")
    print(f'Using Pretrained model {model_file[0]}')
    checkpoint = torch.load(model_file[0])
    teacher_model.load_state_dict(checkpoint['state_dict'])
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model