from copy import deepcopy
import torch
from ptflops import get_model_complexity_info
from torch import nn as nn
from utils.utils import find_module_names, get_module_by_name, set_module_by_name
from models.accm import RsacmBlock, AccmBlock
from models.resnet import ResNet50
import glob

def filter_condition(m: nn.Module):
    allow = False
    if isinstance(m, nn.Conv2d):
        if isinstance(m.kernel_size, tuple):
            for ks_in_dim in m.kernel_size:
                if ks_in_dim > 1:
                    allow = True
        elif isinstance(m.kernel_size, int) and m.kernel_size > 1:
            allow = True
    return allow


def accmize_from(original_model: nn.Module, rsacm_flops_factor: float, number_of_rsacms: int,
                 example_input: torch.Tensor, args):
    training = original_model.training
    original_model.eval()
    model = deepcopy(original_model)
    # find all convolutions that are larger than 1x1
    modules_to_replace = find_module_names(original_model, filter_condition)
    # Set replaced_modules as a model
    model.replaced_modules = modules_to_replace
    # use hooks to save activations for each module into to a dict
    module_activations = {}
    module_id_to_name = {}
    hook_handles = []
    for name in modules_to_replace:
        original_conv = get_module_by_name(model, name)
        module_id_to_name[id(original_conv)] = name
        def save_activations_hook(m, inputs):
            # warning - some modules accept more than 1 input
            # TODO modify to handle such case
            module_activations[module_id_to_name[id(m)]] = inputs[0]

        handle = original_conv.register_forward_pre_hook(save_activations_hook)
        hook_handles.append(handle)
    model(example_input)
    for handle in hook_handles:
        handle.remove()
    # calculate size and replace the selected layers with ACCMs
    for name in modules_to_replace:
        original_conv = get_module_by_name(model, name)
        original_conv_cost, _ = get_model_complexity_info(original_conv, tuple(module_activations[name].size())[1:],
                                                          as_strings=False, print_per_layer_stat=False, verbose=False)
        # calculate C_h for each module, based on rsacm_flops_factor
        candidate_c_h = 1
        rsacm_cost = 0
        while rsacm_cost < rsacm_flops_factor * original_conv_cost:
            candidate_c_h *= 2
            rsacm = RsacmBlock(original_conv, candidate_c_h)
            rsacm_cost, _ = get_model_complexity_info(rsacm, tuple(module_activations[name].size())[1:],
                                                      as_strings=False, print_per_layer_stat=False, verbose=False)
            # print(f'RSACM cost for {name} for C_h {candidate_c_h}: {rsacm_cost}')
        replacement = AccmBlock(original_conv, candidate_c_h, number_of_rsacms, args, rsacm_cost)
        set_module_by_name(model, name, replacement)
        replacement.old_conv_cost = original_conv_cost
        replacement.rsacm_cost = rsacm_cost
        print(f'Replacing {name} - original cost: {original_conv_cost}, RSACM cost: {rsacm_cost}')
    model.train(training)
    original_model.train(training)
    return model

def load_teacher_model(args):
    teacher_model = ResNet50().cuda()
    model_file = glob.glob("./pre-trained/teacher_network/ResNet" + "/*.pth")
    print(f'Using Pretrained model {model_file[0]} for the teacher model')
    checkpoint = torch.load(model_file[0])
    teacher_model.load_state_dict(checkpoint['model_state'])
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model

def load_student_model(args, student_model, teacher_model):
    if args.training_type in ['inference', 'train_gating_networks', 'calculate_complexity', 'eval_gating_networks', 'inference_nas']:
        if args.training_type == 'train_gating_networks':
            model_file = glob.glob(f"./pre-trained/student_network_{args.compresion_rate}/without_gating_networks" + "/*.tar")
        else:
            initialize_gating_networks(student_model, teacher_model, args)
            model_file = glob.glob(f"./pre-trained/student_network_{args.compresion_rate}/with_gating_networks" + "/*.tar")

        print(f'Using Pretrained model {model_file[0]} for the student model')
        checkpoint = torch.load(model_file[0])
        student_model.load_state_dict(checkpoint['state_dict'])

def initialize_gating_networks(student_model, teacher_model, args):
    for index, module_name in enumerate(student_model.replaced_modules):
        get_module_by_name(student_model, module_name).initialize_gating_network(teacher_model.inputs[index][0], args)
        get_module_by_name(student_model, module_name).args.train_gating_networks = True

def get_gating_netowrks_parameters(student_model):
    for index, module_name in enumerate(student_model.replaced_modules):
        gating_networks_parameters += list(get_module_by_name(student_model, module_name).gating_network.parameters())
    return gating_networks_parameters

