from torchvision.transforms import transforms
from torchvision import transforms, datasets

class Dataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform(is_train=True):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
        
        if is_train==True:
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                transforms.GaussianBlur(1, sigma=(0.1, 2.0)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            data_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  
            
        
        return data_transforms

    def get_dataset(self, name, is_train=True):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=is_train,
                                                              transform=self.get_transform(is_train),
                                                              download=True),
                                                                                
        }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise ValueError()
        else:
            return dataset_fn()
