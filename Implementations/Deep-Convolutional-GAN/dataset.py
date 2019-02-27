import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from constants import *


def create_dataloader():

    if DATASET == 'CelebA':
        # Define transform to apply to input images
        trsf = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load dataset from folder
        dataset = datasets.ImageFolder(DATA_PATH, transform=trsf)

    elif DATASET == 'MNIST':
        # Define transform to apply to input images
        trsf = transforms.ToTensor()

        # Load dataset
        train_dataset = datasets.MNIST(DATA_PATH, train=True, transform=trsf, download=True)
        test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=trsf, download=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    elif DATASET == 'CIFAR10':
        # Define transform to apply to input images
        trsf = transforms.ToTensor()

        # Load dataset
        train_dataset = datasets.CIFAR10(DATA_PATH, train=True, transform=trsf, download=True)
        test_dataset = datasets.CIFAR10(DATA_PATH, train=False, transform=trsf, download=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    # Instantiate DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )

    return dataloader
