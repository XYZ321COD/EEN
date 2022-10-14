from torchvision.transforms import transforms
from torchvision import transforms, datasets

class Dataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform(is_train=True):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""        
        if is_train==True:
            data_transforms = transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            data_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  
            
        
        return data_transforms

    def get_dataset(self, name, is_train=True):
        data_set = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=is_train,
                                                              transform=self.get_transform(is_train),
                                                              download=True),
                                                                                
        }
        try:
            dataset_fn = data_set[name]
        except KeyError:
            raise ValueError()
        else:
            return dataset_fn()
