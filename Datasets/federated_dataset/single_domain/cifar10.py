import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset
from Datasets.utils.transforms import DeNormalize
from utils.conf import single_domain_data_path
from PIL import Image
from typing import Tuple
import torchvision.transforms as T

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        #root = '~/miniconda3/lib/python3.12/site-packages/torchvision/datasets$/cifar10'
        #super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()
        self.data = self.dataset.data

        if hasattr(self.dataset, 'classes'):
            self.classes = self.dataset.classes

        if hasattr(self.dataset, 'labels'):
            self.targets = self.dataset.labels

        elif hasattr(self.dataset, 'targets'):
            self.targets = self.dataset.targets

        if isinstance(self.targets, torch.Tensor):
            self.targets = self.targets.numpy()
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.numpy()

    def __build_truncated_dataset__(self):
        dataobj = CIFAR10('~/miniconda3/lib/python3.12/site-packages/torchvision/datasets$/cifar10', self.train, self.transform, self.target_transform, self.download)

        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaCIFAR10(SingleDomainDataset):
    NAME = 'fl_cifar10'
    SETTING = 'label_skew'
    N_CLASS = 10

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)

        self.weak_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.ToTensor()])

        '''
        self.weak_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
        '''

        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])

        '''

        normalization = self.get_normalization_transform()

        self.weak_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalization])

        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalization])

        '''


    def get_data_loaders(self):
        print("Calculating mean and std for FLCIFAR10 dataset...")
        temp_transform = transforms.ToTensor()
        trainset_for_stats = CIFAR10(root='/bsuhome/jonathanauyong/REU/Summer2025REU/data/label_skew', train=True, download=True, transform=temp_transform)
        trainloader_for_stats = torch.utils.data.DataLoader(trainset_for_stats, batch_size=len(trainset_for_stats), shuffle=False)
        images_for_stats, _ = next(iter(trainloader_for_stats)) # Shape: [N, C, H, W]
        mean = images_for_stats.mean(dim=[0, 2, 3])
        std = images_for_stats.std(dim=[0, 2, 3])

        print(f"Calculated Mean for FLCIFAR10: {mean}")
        print(f"Calculated Std Dev for FLCIFAR10: {std}")

        self.calculated_mean = mean
        self.calculated_std = std

        pri_aug = self.cfg.DATASET.aug
        if pri_aug == 'weak':
            train_transform = self.weak_transform
        elif pri_aug == 'strong':
            train_transform = self.strong_transform

        train_dataset = MyCIFAR10('~/miniconda3/lib/python3.12/site-packages/torchvision/datasets$/cifar10/cifar-10-batches-py/data_batch_1', train=True,
                                  download=True, transform=train_transform)
        
        #test_transform = transforms.Compose(
        #    [transforms.ToTensor(), self.get_normalization_transform()])
        test_transform = transforms.Compose(
            [transforms.ToTensor()])
        test_dataset = MyCIFAR10('~/miniconda3/lib/python3.12/site-packages/torchvision/datasets$/cifar10/cifar-10-batches-py/test', train=False,
                               download=True, transform=test_transform)
        #print(test_dataset)
        #print(train_dataset)

        self.partition_label_skew_loaders(train_dataset, test_dataset)

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaCIFAR10.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        return transform
