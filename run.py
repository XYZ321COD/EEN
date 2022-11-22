import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from datasets import Dataset
from enn import ENN
import glob
from utils.models import accmize_from
import pathlib
from utils.models import load_teacher_model, load_student_model
from torch.utils.data import Subset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch HCH')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100', 'mnist', 'svhn' , 'fmnist', 'cifar10kclasses','imagenet10','imagenetdogs'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--num_classes', default=10, type=int,
                    help='number of classes (default: 10)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--training_type', default='train_student',
                    help='training type', choices=['train_student', 'train_gating_networks', 'inference', 'eval_gating_networks',
                         'calculate_complexity', 'inference_nas'])       
parser.add_argument('--number_of_rsacm', default=1, type=int,help="number of RSACMs per ACCM block")
parser.add_argument('--train_gating_networks',  action="store_true",help="Instead of training student model train gating networks")
parser.add_argument('--weight_of_cost_loss', default=0.01, type=float, 
                    metavar='weight_of_cost_loss', help='weight_of_cost_loss', dest='weight_of_cost_loss')
parser.add_argument('--model_type', default='ResNet', help='model arch', choices=['ResNet'])
parser.add_argument('--compresion_rate', default=0.05, type=float)

def main():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        # cudnn.deterministic = True
        # cudnn.benchmark = False
        args.gpu_index = 0 
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = Dataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, is_train=True, args=args)
    valid_dataset = dataset.get_dataset(args.dataset_name, is_train=False, args=args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    teacher_model = load_teacher_model(args)

    example_input = next(iter(train_loader))[0][:1]
    student_model = accmize_from(teacher_model, args.compresion_rate, args.number_of_rsacm, example_input.cuda(), args)
    optimizer = torch.optim.Adam(student_model.parameters(), args.lr, weight_decay=0)

    teacher_model.replaced_modules = student_model.replaced_modules
    teacher_model.register_all_hooks()
    student_model.register_all_hooks()
    teacher_model(example_input.cuda())

    load_student_model(args, student_model, teacher_model) 
    del teacher_model

    with torch.cuda.device(args.gpu_index):
        enn = ENN(model=student_model, optimizer=optimizer, args=args)

        if args.training_type =='calculate_complexity':
            enn.calculated_average_complexity(valid_loader)

        if args.training_type == 'inference':
            enn.inference(train_loader, valid_loader)

        if args.training_type == 'train_student':
            enn.train(train_loader, valid_loader)

        if args.training_type == 'train_gating_networks':
            enn.train_gating_networks(train_loader, valid_loader)

        if args.training_type == 'eval_gating_networks':
            enn.eval_gating_networks(valid_loader)

        if args.training_type == 'inference_nas':
            enn.inference_nas(valid_loader)



if __name__ == "__main__":
    main()
