import os
import shutil
from functools import reduce
from typing import Callable
from tqdm import tqdm
import torch
import yaml
from torch import nn as nn
from ptflops import get_model_complexity_info
import logging

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def find_module_names(module: nn.Module, filter: Callable[[nn.Module], bool]):
    found_names = []
    for name, module in module.named_modules():
        if filter(module):
            found_names.append(name)
    return found_names


def get_module_by_name(module: nn.Module, name: str):
    names = name.split(sep='.')
    return reduce(getattr, names, module)


def set_module_by_name(module: nn.Module, name: str, replacement: nn.Module):
    names = name.split(sep='.')
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], replacement)




def eval(valid_loader, model, args):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    return 100 * correct // total


def eval_partial(valid_loader, model, number_of_rscam_to_eval, args):
    correct = 0
    total = 0
    model.override_forward_for_accm_blocks(number_of_rscam_to_eval)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    return 100 * correct // total



def calculated_complexity(student_model, teacher_model):
    macs, _ = get_model_complexity_info(student_model, (3, 32, 32), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
    gflops = 2*macs
    logging.debug(f"Computational complexity for student model: {gflops} FLOPS")

    macs, _ = get_model_complexity_info(teacher_model, (3, 32, 32), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
    gflops = 2*macs
    logging.debug(f"Computational complexity for teacher model: {gflops} FLOPS") 

