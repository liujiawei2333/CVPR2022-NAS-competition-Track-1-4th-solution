from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from prefetch_generator import BackgroundGenerator
import pandas as pd
import torch
import numpy as np


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


'''Dataset used to train the supernet'''
class DatasetLoader(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        the formation of txt file: image path, label
        """
        with open(txt_file, "r") as f:
            self.dataset_lines = f.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.dataset_lines)

    def __getitem__(self, idx):
        line = self.dataset_lines[idx].strip()
        img_path, label = line.split(",")
        label = int(label)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


'''Dataset used to test the subnet'''
class DatasetLoader_during_test(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        the formation of txt file: image path, label
        """
        with open(txt_file, "r") as f:
            self.dataset_lines = f.readlines()
        self.transform = transform
        self.mean = torch.as_tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=torch.device("cpu"))[:, None,
                    None]
        self.std = torch.as_tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=torch.device("cpu"))[:, None,
                   None]

    def __len__(self):
        return len(self.dataset_lines)

    def __getitem__(self, idx):
        line = self.dataset_lines[idx].strip()
        img_path, label = line.split(",")
        label = int(label)

        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image.float().div(255)
        image = image.sub_(self.mean).div_(self.std)

        if self.transform:
            image = self.transform(image)
        return image, label


'''DataLoader used to test the subnet'''
def get_val_dataset_loader_imagenet(test_set, args, common_args):
    if test_set == 1:  # Full test set
        val_dataset = DatasetLoader_during_test(common_args.imagenet_val_all_csv, transform=None)
    elif test_set == 0:  # Partial test set
        val_dataset = DatasetLoader_during_test(common_args.imagenet_val_csv, transform=None)

    val_loader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )

    return val_loader


'''DataLoader used to train the supernet'''
def get_dataset_loader_imagenet(args, common_args):
    train_process = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomRotation(15, resample=False, expand=False, center=None),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    val_process = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    train_dataset = DatasetLoader(common_args.imagenet_train_csv, transform=train_process)
    val_dataset = DatasetLoader(common_args.imagenet_val_csv, transform=val_process)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=getattr(common_args, 'drop_last', True),
        num_workers=common_args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    val_loader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=common_args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler


'''DataLoader used to train the subnet stand alone'''
def get_dataset_loader_stand_alone(args, common_args):
    train_process = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    val_process = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    train_dataset = DatasetLoader(common_args.imagenet_train_csv, transform=train_process)
    val_dataset = DatasetLoader(common_args.imagenet_val_csv, transform=val_process)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=getattr(common_args, 'drop_last', True),
        num_workers=common_args.data_loader_workers_per_gpu,
        pin_memory=True,
    )

    val_loader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=common_args.data_loader_workers_per_gpu,
        drop_last=False,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler
