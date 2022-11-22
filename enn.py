import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.utils import *
from utils.models import load_teacher_model, initialize_gating_networks, get_gating_netowrks_parameters
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt 
import numpy as np 

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

     
    def train(self, train_loader, valid_loader):
        """
        Train student networks
        """
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
                teacher_outputs = self.teacher_model(images)
                loss_distill = 0
                for index, module_name in enumerate(self.model.replaced_modules):
                    get_module_by_name(self.model, module_name)(self.teacher_model.inputs[index][0])
                    for rsacm_block_output in get_module_by_name(self.model, module_name).x_list:
                        loss_distill += self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index])
                loss_distill.backward()
                self.optimizer.step()
                
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('Loss', loss_distill, global_step=n_iter)            
                n_iter += 1
                
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


    def train_gating_networks(self, train_loader, valid_loader):
        """
        Train gating networks of pre-trained student network
        """

        accuracy_student = eval(valid_loader, self.model, self.args)
        logging.debug(f"Before Training gating networks student_model: \t Accuracy {accuracy_student}")
        gating_networks_parameters = []
        ## Initialize the gating networks
        initialize_gating_networks(self.model, self.teacher_model, self.args)
        gating_networks_parameters =  get_gating_netowrks_parameters(self.model)
        gating_networks_optimizer = torch.optim.Adam(gating_networks_parameters, lr=0.001, weight_decay=1e-4)

        torch.autograd.set_detect_anomaly(True)
        save_config_file(self.writer.log_dir, self.args)

        for epoch_counter in range(self.args.epochs):
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                output = self.model(images)
                loss_cost_classification = self.criterion(output, labels)
                loss_cost_gn = 0
                gating_networks_optimizer.zero_grad()
                for index, module_name in enumerate(self.model.replaced_modules):
                    loss_cost_gn += get_module_by_name(self.model, module_name).loss_cost_gn
                loss_summ = loss_cost_classification + self.args.weight_of_cost_loss*(loss_cost_gn)
                if i % 50 == 0:
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


    def eval_gating_networks(self, valid_loader):
        """
        Checks how many rsacms are used per example in dataset
        """
        histograms_for_layers = {module_name: [] for module_name in self.model.replaced_modules}
        del self.teacher_model
        for i, (images, labels) in enumerate(tqdm(valid_loader)):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            self.model(images)
            for index, module_name in enumerate(self.model.replaced_modules):
                histograms_for_layers[module_name] += torch.argmax(get_module_by_name(self.model, module_name).gating_network_output_detached, dim=1).tolist()
        visualize_loss_diff(histograms_for_layers)


    def inference_nas(self, valid_loader):
        """
        Checks the complexity and accuracy for the student model with respect to different number of rsacm activate in each layer.
        """
        if self.args.train_gating_networks:
            accuracy = eval(valid_loader, self.model, self.args)
            print(accuracy)
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
        """
        Visualize difference between the activation of the teacher network and student network
        """
        def calculated_losses_per_layers(dataset):
            with torch.no_grad():
                losses_per_layers_student = {module_name: [[] for i in range(0,self.args.number_of_rsacm)] for module_name in self.model.replaced_modules}
                for i, (images, labels) in enumerate(tqdm(dataset)):
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        self.teacher_model(images)
                        loss_distill = 0
                        for index, module_name in enumerate(self.model.replaced_modules):
                            get_module_by_name(self.model, module_name)(self.teacher_model.inputs[index][0])
                            for rsacm_index, rsacm_block_output in enumerate(get_module_by_name(self.model, module_name).x_list):
                                loss_distill += self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index])
                                losses_per_layers_student[module_name][rsacm_index].append(self.criterion_distilation(rsacm_block_output, self.teacher_model.activation[index]))
                losses_per_layers_student_mean = {module_name: [(sum(losses_per_layers_student[module_name][i]) / len(losses_per_layers_student[module_name][i])).item() for i in range(0, self.args.number_of_rsacm)] for module_name in self.model.replaced_modules}
                return losses_per_layers_student_mean

        losses_per_layers_student_mean_valid, losses_per_layers_student_mean_train = calculated_losses_per_layers(valid_loader), calculated_losses_per_layers(train_loader)

        generate_loss_histograms(self.args, losses_per_layers_student_mean_valid, losses_per_layers_student_mean_train)    

        
    def calculated_average_complexity(self, valid_loader):
        """
        Calculates complexity of student network with gating networks and compares it to complexity of teacher network
        """
        complexity_of_teacher = 0
        complexity_of_student = 0
        for i, (images, labels) in enumerate(tqdm(valid_loader)):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            self.model(images)
            for index, module_name in enumerate(self.model.replaced_modules):
                output = torch.argmax(get_module_by_name(self.model, module_name).gating_network_output_detached, dim=1)
                complexity_of_student += (torch.mean(output.float()) + 1 ) * get_module_by_name(self.model, module_name).rsacm_cost
                complexity_of_teacher +=  get_module_by_name(self.model, module_name).old_conv_cost
        print(f"Percentage {complexity_of_student/complexity_of_teacher}")
        print(f"Accuracy of the student model {eval(valid_loader, self.model, self.args)}")  
        print(f"Accuracy of the teacher model {eval(valid_loader, self.teacher_model, self.args)}")        