from copy import deepcopy
import torch
from ptflops import get_model_complexity_info
from torch import nn as nn
from utils.utils import find_module_names, get_module_by_name, set_module_by_name
from models.accm import RsacmBlock, AccmBlock

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
                 example_input: torch.Tensor):
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
        replacement = AccmBlock(original_conv, candidate_c_h, number_of_rsacms)
        set_module_by_name(model, name, replacement)
        print(f'Replacing {name} - original cost: {original_conv_cost}, RSACM cost: {rsacm_cost}')
    model.train(training)
    original_model.train(training)
    return model