import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, save_checkpoint
from resnet import ResNet
import glob
from copy import copy
from ptflops import get_model_complexity_info


class ENN(object):

    def calculated_complexity(self):
        macs, params = get_model_complexity_info(self.model, (3, 32, 32), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
        gflops = 2*macs
        logging.debug(f"Computational complexity for student model: {gflops} FLOPS")

        macs, params = get_model_complexity_info(self.teacher_model, (3, 32, 32), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
        gflops = 2*macs
        logging.debug(f"Computational complexity for teacher model: {gflops} FLOPS") 


    def load_teacher_model(self, args):
        self.args_copy = copy(args)
        self.args_copy.accm = False
        
        teacher_model = ResNet(base_model=self.args_copy.arch, out_dim=self.args_copy.num_classes, pretrained=self.args_copy.pretrained, args=self.args_copy).cuda()
        model_file = glob.glob("./runs/Sep30_23-18-33_DESKTOP-300CBSN" + "/*.tar")
        print(f'Using Pretrained model {model_file[0]}')
        checkpoint = torch.load(model_file[0])
        teacher_model.load_state_dict(checkpoint['state_dict'])
        for param in teacher_model.parameters():
            param.requires_grad = False
        return teacher_model

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.criterion_distilation = torch.nn.MSELoss().to(self.args.device)
        self.teacher_model = self.load_teacher_model(self.args)
        self.teacher_model.layers_to_replace()
        self.teacher_model.register_all_hooks()
        self.calculated_complexity()

    def eval(self, valid_loader, model):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        return 100 * correct // total


    def eval_partial(self, valid_loader, model, number_of_rscam_to_eval):
        correct = 0
        total = 0
        self.model.backbone.conv1.override_forward(number_of_rscam_to_eval)
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        return 100 * correct // total


    def train(self, train_loader):
        print(self.model)
        torch.autograd.set_detect_anomaly(True)
        # scaler = GradScaler()
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        accuracy_student = self.eval(train_loader, self.model)
        accuracy_teacher = self.eval(train_loader, self.teacher_model)
        logging.debug(f"Before Training student_model: \t Accuracy {accuracy_student}")
        logging.debug(f"Before Training teacher_model: \t Accuracy {accuracy_teacher}")


        for epoch_counter in range(self.args.epochs):
            for i, (images, labels) in enumerate(tqdm(train_loader)):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # Changes number of rsacm 
                if self.args.train_rsacm:
                    random_rsacm = torch.randint(1, self.model.backbone.conv1.number_of_rsacm+1, (1,))
                    self.model.backbone.conv1.override_forward(random_rsacm)
                loss_distill = 0
                outputs = self.model(images)
                teacher_outputs = self.teacher_model(images)
                for key, value in self.model.activation.items():
                    loss_distill += self.criterion_distilation(self.model.activation[key], self.teacher_model.activation[key])
                self.optimizer.zero_grad()
                loss_distill.backward()
                self.optimizer.step()

                
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('Loss', loss_distill, global_step=n_iter)            
                n_iter += 1
                
            if epoch_counter >= 10:
                self.scheduler.step()
            if self.args.train_rsacm:
                self.model.backbone.conv1.override_forward(self.args.number_of_rsacm)
            accuracy = self.eval(train_loader, self.model)
            logging.debug(f"Epoch: {epoch_counter}\Loss: {loss_distill}\t")
            logging.debug(f"Epoch: {epoch_counter}\t Accuracy {accuracy}")
        for i in range(1, self.args.number_of_rsacm+1):
            accuracy_partial = self.eval_partial(train_loader, self.model, i)
            macs, params = get_model_complexity_info(self.model, (3, 32, 32), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
            gflops = 2*macs
            logging.debug(f"Computational complexity for student model: {gflops} FLOPS when using {i} number of rsacm: Accuracy {accuracy_partial}")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        