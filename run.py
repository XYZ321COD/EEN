import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from datasets import Dataset
from models.resnet import ResNet50
from enn import ENN
import glob
from utils.models import accmize_from
import pathlib

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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--num_classes', default=10, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--pretrained', action='store_true', help='use pretrained network')
parser.add_argument('--accm', action='store_true', help='use accm layer')
parser.add_argument('--save_point', default=".", type=str, help="Path to .pth ")
parser.add_argument('--load_model', default=False, action="store_true",help="Use pretrained model")
parser.add_argument('--save_feature_map', action="store_true",help="save_feature_map")
parser.add_argument('--h_channels', default=32, type=int,help="h_channels")
parser.add_argument('--number_of_rsacm', default=1, type=int,help="number_of_rsacm")
parser.add_argument('--train_rsacm', action="store_true",help="train_rsacm")
parser.add_argument('--distill_type', default='',
                    help='distill type', choices=['per_accm', 'per_rsacm'])
parser.add_argument('--inference', action="store_true",help="train_rsacm")
parser.add_argument('--load_student', action="store_true",help="train_rsacm")
parser.add_argument('--train_gating_networks',  action="store_true",help="train_rsacm")
parser.add_argument('--weight_of_cost_loss', default=1, type=float, 
                    metavar='weight_of_cost_loss', help='weight_of_cost_loss', dest='weight_of_cost_loss')


def main():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
        args.gpu_index = 0 
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = Dataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, is_train=True)

    valid_dataset = dataset.get_dataset(args.dataset_name, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    teacher_model = ResNet50()
    print(teacher_model)
    model_file = glob.glob("./pre-trained/resnet_bartek" + "/*.pth")
    print(f'Using Pretrained model {model_file[0]} for the teacher model')
    checkpoint = torch.load(model_file[0])
    teacher_model.load_state_dict(checkpoint['model_state'])
    
    example_input = next(iter(train_loader))[0][:1]
    student_model = accmize_from(teacher_model, 0.25, args.number_of_rsacm, example_input, args)
    optimizer = torch.optim.Adam(student_model.parameters(), args.lr, weight_decay=0)

    if args.load_student:
        model_file = glob.glob("./pre-trained/Oct13_17-05-07_DESKTOP-300CBSN" + "/*.tar")
        print(f'Using Pretrained model {model_file[0]} for the teacher model')
        checkpoint = torch.load(model_file[0])
        student_model.load_state_dict(checkpoint['state_dict'])
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        enn = ENN(model=student_model, optimizer=optimizer, args=args)
        enn.train_gating_networks(train_loader, valid_loader)
        if args.inference:
            enn.inference(train_loader, valid_loader)
        else:
            enn.train(train_loader, valid_loader)


if __name__ == "__main__":
    main()
