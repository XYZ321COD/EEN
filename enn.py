import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, save_checkpoint

class ENN(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        logging.debug(f"Running with accm: {self.args.accm}")
        
    def eval(self, valid_loader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        return 100 * correct // total


    def train(self, train_loader):
        print(self.model)
        torch.autograd.set_detect_anomaly(True)
        scaler = GradScaler()
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        for epoch_counter in range(self.args.epochs):
            for i, (images, labels) in enumerate(tqdm(train_loader)):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update() 

                
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('Loss', loss, global_step=n_iter)            
                n_iter += 1
                
            if epoch_counter >= 10:
                self.scheduler.step()
            accuracy = self.eval(train_loader)
            logging.debug(f"Epoch: {epoch_counter}\Loss: {loss}\t")
            logging.debug(f"Epoch: {epoch_counter}\t Accuracy {accuracy}")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        