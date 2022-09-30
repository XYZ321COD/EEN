from torchvision.transforms import transforms
from torchvision import transforms, datasets

class Dataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform():
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        
        data_transforms = transforms.Compose([
                                              transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        return data_transforms

    def get_dataset(self, name):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=self.get_transform(),
                                                              download=True),
                                                              

        'imagenet10': lambda: datasets.ImageNet('./datasets', split='val', transform=[transforms.Resize((224,224)),transforms.ToTensor()]),
                        
        }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise ValueError()
        else:
            return dataset_fn()
