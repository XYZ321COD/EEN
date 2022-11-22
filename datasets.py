from torchvision.transforms import transforms
from torchvision import transforms, datasets
import glob
class Dataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform(is_train=True, name=""):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""    

        if is_train==True:
            if name == 'cifar10':
                data_transforms = transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            else:
                data_transforms = transforms.Compose([
                                                transforms.RandomSizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        else:
            if name == 'cifar10':
                data_transforms = transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  
            else:
                data_transforms = transforms.Compose([
                                                    transforms.RandomSizedCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

            
        
        return data_transforms

    def get_dataset(self, name, is_train=True, args={}):

        split = 'train' if is_train else 'val'
        
        data_set = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=is_train,
                                                              transform=self.get_transform(is_train, name),
                                                              download=True),
                    
                    'imagenet10': lambda: datasets.ImageNet('../DS-Net/Datasets', split=split,
                                                              transform=self.get_transform(is_train, name),
                                                              ),

                                                                                
        }
        try:
            dataset_fn = data_set[name]
        except KeyError:
            raise ValueError()
        else:
            return dataset_fn()
