import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.utils import get_module_by_name, save_config_file, save_checkpoint, eval, eval_partial, calculated_complexity
from models.resnet import load_teacher_model
from models.gating_network import Gating_Network, initialized_gating_networks
import glob
from copy import copy
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt 
import numpy as np 
from models.accm import AccmBlock

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



    def inference_nas(self, train_loader, valid_loader):
        for idx, value in enumerate(self.model.replaced_modules):
            for number_of_rsacm in range(1,self.args.number_of_rsacm+1):
                with torch.no_grad():
                    get_module_by_name(self.model, value).override_forward(number_of_rsacm)
                    accuracy = eval(valid_loader, self.model, self.args)
                    macs, _ = get_model_complexity_info(self.model, (3, 32, 32), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
                    gflops = 2*macs
                    logging.debug(f"Computational complexity for student model: {gflops} FLOPS when using {number_of_rsacm} number of rsacm in layer {value}: Accuracy {accuracy}")



    def inference(self, train_loader, valid_loader):
        with torch.no_grad():
            losses_per_layers_student_valid = {module_name: [[] for i in range(0,self.args.number_of_rsacm)] for module_name in self.model.replaced_modules}
            for i, (images, labels) in enumerate(tqdm(valid_loader)):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    self.teacher_model(images)
                    loss_distill = 0
                    for index, module_name in enumerate(self.model.replaced_modules):
                        get_module_by_name(self.model, module_name)(self.teacher_model.inputs[index][0])
                        for rsacm_index, rsacm_block_output in enumerate(get_module_by_name(self.model, module_name).x_list):
                            loss_distill += self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index])
                            losses_per_layers_student_valid[module_name][rsacm_index].append(self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index]))
            # print(losses_per_layers_student)
            losses_per_layers_student_mean_valid = {module_name: [(sum(losses_per_layers_student_valid[module_name][i]) / len(losses_per_layers_student_valid[module_name][i])).item() for i in range(0, self.args.number_of_rsacm)] for module_name in self.model.replaced_modules}

            losses_per_layers_student_train = {module_name: [[] for i in range(0,self.args.number_of_rsacm)] for module_name in self.model.replaced_modules}
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    self.teacher_model(images)
                    loss_distill = 0
                    for index, module_name in enumerate(self.model.replaced_modules):
                        get_module_by_name(self.model, module_name)(self.teacher_model.inputs[index][0])
                        for rsacm_index, rsacm_block_output in enumerate(get_module_by_name(self.model, module_name).x_list):
                            loss_distill += self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index])
                            losses_per_layers_student_train[module_name][rsacm_index].append(self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index]))
            # print(losses_per_layers_student)
            losses_per_layers_student_mean_train = {module_name: [(sum(losses_per_layers_student_train[module_name][i]) / len(losses_per_layers_student_train[module_name][i])).item() for i in range(0, self.args.number_of_rsacm)] for module_name in self.model.replaced_modules}
            for module_name in losses_per_layers_student_mean_valid:
                # Make a random dataset:
                rsacms = range(1, self.args.number_of_rsacm+1)
                loss_value_valid = losses_per_layers_student_mean_valid[module_name]
                loss_value_train = losses_per_layers_student_mean_train[module_name]
                y_pos = np.arange(len(rsacms))
                width = 0.3       

                # Create bars
                plt.bar(y_pos, loss_value_valid, width, label='valid')
                plt.bar(y_pos+width, loss_value_train, width, label='train')

                # Create names on the x-axis
                plt.legend(loc='best')
                plt.xticks(y_pos, rsacms)
                plt.title(f'MSELoss in {module_name}')
                plt.plot()
                plt.xlabel('Number of RSACM')
                plt.ylabel('MSELoss value')
                plt.savefig(f'./graphs/{module_name}.png')
                plt.clf()


    def train_gating_networks(self, train_loader, valid_loader):

        accuracy_student = eval(valid_loader, self.model, self.args)
        logging.debug(f"Before Training gating networks student_model: \t Accuracy {accuracy_student}")
        optimizers_list = []
        ## Initialize the gating networks
        for index, module_name in enumerate(self.model.replaced_modules):
            get_module_by_name(self.model, module_name).initialize_gating_network(self.teacher_model.inputs[index][0], self.args)
            get_module_by_name(self.model, module_name).args.train_gating_networks = True
            optimizers_list += list(get_module_by_name(self.model, module_name).gating_network.parameters())
        gating_networks_optimizer = torch.optim.Adam(optimizers_list)

        torch.autograd.set_detect_anomaly(True)
        save_config_file(self.writer.log_dir, self.args)
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            if i == 300:
                break
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            output = self.model(images)
            loss_cost_classification = self.criterion(output, labels)
            loss_cost_gn = 0
            gating_networks_optimizer.zero_grad()
            for index, module_name in enumerate(self.model.replaced_modules):
                loss_cost_gn += get_module_by_name(self.model, module_name).loss_cost_gn
            loss_summ = loss_cost_classification + self.args.weight_of_cost_loss*(loss_cost_gn)
            if i % 100 == 0:
                print(f'CE {loss_cost_classification} and cost {loss_cost_gn / len((self.model.replaced_modules))}')
            loss_summ.backward()
            gating_networks_optimizer.step()
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        accuracy_student = eval(valid_loader, self.model, self.args)
        logging.debug(f"Before Training gating networks student_model: \t Accuracy {accuracy_student}")
     
    def train(self, train_loader, valid_loader):
        optimizers_list = []
        ## Initialize the gating networks
        for index, module_name in enumerate(self.model.replaced_modules):
            get_module_by_name(self.model, module_name).initialize_gating_network(self.teacher_model.inputs[index][0], self.args)
            optimizers_list += list(get_module_by_name(self.model, module_name).gating_network.parameters())
        gating_networks_optimizer = torch.optim.Adam(optimizers_list)

        torch.autograd.set_detect_anomaly(True)
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        accuracy_student = eval(valid_loader, self.model, self.args)
        accuracy_teacher = eval(valid_loader, self.teacher_model, self.args)
        logging.debug(f"Before Training student_model: \t Accuracy {accuracy_student}")
        logging.debug(f"Before Training teacher_model: \t Accuracy {accuracy_teacher}")

        for epoch_counter in range(self.args.epochs):
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.optimizer.zero_grad()
                gating_networks_optimizer.zero_grad()
                output = None
                teacher_outputs = self.teacher_model(images)
                if self.args.distill_type == 'per_accm':
                    loss_distill = 0
                    for key, value in self.model.activation.items():
                        loss_distill += self.criterion_distilation(self.model.activation[key], self.teacher_model.activation[key])

                if self.args.distill_type == 'per_rsacm':
                    loss_distill = 0
                    loss_cost_gn = 0 
                    for index, module_name in enumerate(self.model.replaced_modules):
                        if self.args.train_gating_networks:
                            if index == 0:
                                get_module_by_name(self.model, module_name)(self.teacher_model.inputs[index][0])
                                gating_network_output =  get_module_by_name(self.model, module_name).gating_network(self.teacher_model.inputs[index][0])
                                gating_network_output_index = torch.argmax(gating_network_output, dim=1)
                                costs = torch.tensor([10485760.0, 2*10485760.0, 3*10485760.0, 4*10485760.0, 5*10485760.0, 6*10485760.0, 7*10485760.0, 8*10485760.0]).cuda()
                                lists_of_outputs = []
                                for index2, elements in enumerate(gating_network_output_index):
                                    lists_of_outputs.append(get_module_by_name(self.model, module_name).x_list_tensor.permute(1,0,2,3,4)[index2][elements])
                                output = torch.stack(lists_of_outputs)
                                loss_cost_gn += torch.mean((gating_network_output * costs)) * 8
                            else:
                                get_module_by_name(self.model, module_name)(self.teacher_model.inputs[index][0])
                                gating_network_output =  get_module_by_name(self.model, module_name).gating_network(self.teacher_model.inputs[index][0])
                                gating_network_output_index = torch.argmax(gating_network_output, dim=1)
                                costs = torch.tensor([10485760.0, 2*10485760.0, 3*10485760.0, 4*10485760.0, 5*10485760.0, 6*10485760.0, 7*10485760.0, 8*10485760.0]).cuda()
                                lists_of_outputs = []
                                for index2, elements in enumerate(gating_network_output_index):
                                    lists_of_outputs.append(get_module_by_name(self.model, module_name).x_list_tensor.permute(1,0,2,3,4)[index2][elements])
                                output = torch.stack(lists_of_outputs)
                                loss_cost_gn += torch.mean((gating_network_output * costs)) * 8
                        else:
                            get_module_by_name(self.model, module_name)(self.teacher_model.inputs[index][0])
                            for rsacm_block_output in get_module_by_name(self.model, module_name).x_list:
                                loss_distill += self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index])
                loss_cost_gn.backward()
                # loss_distill.backward()
                # self.optimizer.step()
                gating_networks_optimizer.step()
                
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('Loss', loss_distill, global_step=n_iter)            
                n_iter += 1
                
            # if self.args.train_rsacm:
            #     self.model.override_forward_for_accm_blocks(self.args.number_of_rsacm)
            accuracy = eval(valid_loader, self.model, self.args)
            logging.debug(f"Epoch: {epoch_counter}\Loss: {loss_distill}\t")
            logging.debug(f"Epoch: {epoch_counter}\t Accuracy {accuracy}")
        for i in range(1, self.args.number_of_rsacm+1):
            accuracy_partial = eval_partial(valid_loader, self.model, i, self.args)
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
        