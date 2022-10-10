import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.utils import get_module_by_name, save_config_file, save_checkpoint, eval, eval_partial, calculated_complexity
from models.resnet import load_teacher_model
import glob
from copy import copy
from ptflops import get_model_complexity_info


class ENN(object):


    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.criterion_distilation = torch.nn.MSELoss().to(self.args.device)
        self.teacher_model = load_teacher_model(self.args)
        self.teacher_model.replaced_modules = self.model.replaced_modules
        self.teacher_model.register_all_hooks()
        self.model.register_all_hooks()
        calculated_complexity(self.model, self.teacher_model)
        self.teacher_model.eval()
        self.model.eval()

    def train(self, train_loader):
        torch.autograd.set_detect_anomaly(True)
        # scaler = GradScaler()
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        accuracy_student = eval(train_loader, self.model, self.args)
        accuracy_teacher = eval(train_loader, self.teacher_model, self.args)
        logging.debug(f"Before Training student_model: \t Accuracy {accuracy_student}")
        logging.debug(f"Before Training teacher_model: \t Accuracy {accuracy_teacher}")

        for epoch_counter in range(self.args.epochs):
            for i, (images, labels) in enumerate(tqdm(train_loader)):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # Changes number of rsacm 
                if self.args.train_rsacm:
                    random_rsacm = torch.randint(1, self.args.number_of_rsacm+1, (1,))
                    self.model.override_forward_for_accm_blocks(random_rsacm)
                outputs = self.model(images)
                self.optimizer.zero_grad()

                teacher_outputs = self.teacher_model(images)
                if self.args.distill_type == 'per_accm':
                    loss_distill = 0
                    for key, value in self.model.activation.items():
                        # print(self.model.activation[key].shape)
                        # print(self.teacher_model.activation[key].shape)
                        loss_distill += self.criterion_distilation(self.model.activation[key], self.teacher_model.activation[key])

                if self.args.distill_type == 'per_rsacm':
                    loss_distill = 0
                    for index, module_name in enumerate(self.model.replaced_modules):
                        for rsacm_block_output in get_module_by_name(self.model, module_name).x_list:
                            # print(rsacm_block_output.shape)
                            # print(self.teacher_model.activation[index].shape)
                            loss_distill += self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index])

                loss_distill.backward()
                self.optimizer.step()

                
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('Loss', loss_distill, global_step=n_iter)            
                n_iter += 1
                
            # if self.args.train_rsacm:
            #     self.model.override_forward_for_accm_blocks(self.args.number_of_rsacm)
            accuracy = eval(train_loader, self.model, self.args)
            logging.debug(f"Epoch: {epoch_counter}\Loss: {loss_distill}\t")
            logging.debug(f"Epoch: {epoch_counter}\t Accuracy {accuracy}")
        for i in range(1, self.args.number_of_rsacm+1):
            accuracy_partial = eval_partial(train_loader, self.model, i, self.args)
            macs, _ = get_model_complexity_info(self.model, (3, 32, 32), as_strings=False,
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
        