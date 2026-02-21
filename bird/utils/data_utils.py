import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100


img_norm_means = (0.4914, 0.4822, 0.4465)
img_norm_stds = (0.2023, 0.1994, 0.2010)

dataset_nclasses = {
    "cifar10": 10,
    "cifar10-c": 10,
    "cifar100": 100,
    "cifar100-c": 100,
    "tin": 200,
    "tin-c": 200,
}

cifar10_default_train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(img_norm_means, img_norm_stds),
])

cifar10_default_val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(img_norm_means, img_norm_stds),
])

cifar10_c_default_train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(img_norm_means, img_norm_stds),
])

cifar10_c_default_val_transforms = transforms.Compose([
    transforms.Normalize(img_norm_means, img_norm_stds),
])


class CorruptedImages(Dataset):

    train_corruptions = [
        "gaussian_noise", 
        "shot_noise", 
        "impulse_noise", 
        "defocus_blur", 
        "glass_blur",
        "motion_blur", 
        "zoom_blur", 
        "snow", 
        "frost", 
        "fog", 
        "brightness", 
        "contrast", 
        "elastic_transform", 
        "pixelate", 
        "jpeg_compression"
    ]

    test_corruptions = [
        "spatter", "saturate", "speckle_noise", "gaussian_blur"
    ]

    def __init__(
        self, 
        data_root,
        num_samples_per_corruption,
        corruption_types,
        corruption_severities=[1,2,3,4,5],
        img_transform=None,
    ):
        self.num_samples_per_corruption = num_samples_per_corruption
        data_shape = (self.num_samples_per_corruption*6, 3, 32, 32)

        self.corrpution_severities = corruption_severities
        filtered_indices = []
        for sev in self.corrpution_severities:
            filtered_indices += list(
                np.arange(0, self.num_samples_per_corruption) + (sev*self.num_samples_per_corruption)
            )
        filtered_indices = np.array(filtered_indices)

        labels = np.memmap(
            os.path.join(data_root, "all_labels.memmap"),
            dtype=np.int64,
            mode="r+",
            shape=(data_shape[0],),
        )
        labels = labels[filtered_indices]

        self.corruption_types = corruption_types
        self.corrupted_img_data = []
        self.labels = []
        for corruption_type in self.corruption_types:
            data = np.memmap(
                os.path.join(data_root, f"{corruption_type}.memmap"),
                dtype=np.float32,
                mode="r+",
                shape=data_shape,
            )
            data = data[filtered_indices]
            self.corrupted_img_data.append(data)
            self.labels.append(labels)

        self.corrupted_img_data = np.vstack(self.corrupted_img_data)
        self.labels = np.hstack(self.labels)

        self.img_transform = img_transform if img_transform is not None else (lambda x: x)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(
            self.corrupted_img_data[idx]
        )
        img = self.img_transform(img)
        label = self.labels[idx]
        return img, label


def get_cifar10_train(cifar10_data_root, img_tranform=cifar10_default_train_transforms, batch_size=128, num_workers=8):
    """
    Gets dataloader for clean cifar10 train data (32x32)

    Args:
        cifar10_data_root (str): path to data where cifar10 data is saved
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset=CIFAR10(
            root=cifar10_data_root, train=True, transform=img_tranform, download=False
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

def get_cifar10_val(cifar10_data_root, img_tranform=cifar10_default_val_transforms, batch_size=128, num_workers=4):
    """
    Gets dataloader for clean cifar10 val data (32x32)

    Args:
        cifar10_data_root (str): path to data where cifar10 data is saved
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset=CIFAR10(
            root=cifar10_data_root, train=False, transform=img_tranform, download=False
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

def get_cifar10_c_train(
    cifar10_c_data_root,
    corruption_types, 
    corruption_severities=[1,2,3,4,5], 
    img_transform=cifar10_c_default_train_transforms, 
    batch_size=128, 
    num_workers=8
):
    """
    Gets dataloader for corrupted cifar10 training data (32x32)

    Args:
        cifar10_c_data_root (str): path to data where corrupted cifar10 data is saved
        corruption_types (list[str]): list of corruption types to use
        corruption_severities (list[int]): list corruption severities 0-5 (0 means clean data)
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(cifar10_c_data_root, "train_memmap"), 
        num_samples_per_corruption=50000, # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=corruption_types, 
        corruption_severities=corruption_severities, 
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

def get_cifar10_c_val(
    cifar10_c_data_root,
    corruption_types, 
    corruption_severities=[1,2,3,4,5], 
    img_transform=cifar10_c_default_val_transforms,
    batch_size=128,
    num_workers=4
):
    """
    Gets dataloader for corrupted cifar10 val data (32x32)

    Args:
        cifar10_c_data_root (str): path to data where corrupted cifar10 data is saved
        corruption_types (list[str]): list of corruption types to use
        corruption_severities (list[int]): list corruption severities 0-5 (0 means clean data)
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(cifar10_c_data_root, "test_memmap"), 
        num_samples_per_corruption=10000,  # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=corruption_types, 
        corruption_severities=corruption_severities, 
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

def get_cifar100_train(cifar100_data_root, img_tranform=cifar10_default_train_transforms, batch_size=128, num_workers=8):
    """
    Gets dataloader for clean cifar100 train data (32x32)

    Args:
        cifar100_data_root (str): path to data where cifar100 data is saved
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset=CIFAR100(
            root=cifar100_data_root, train=True, transform=img_tranform, download=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

def get_cifar100_val(cifar100_data_root, img_tranform=cifar10_default_val_transforms, batch_size=128, num_workers=4):
    """
    Gets dataloader for clean cifar100 val data (32x32)

    Args:
        cifar100_data_root (str): path to data where cifar100 data is saved
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset=CIFAR100(
            root=cifar100_data_root, train=False, transform=img_tranform, download=True
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

def get_cifar100_c_train(
    cifar100_c_data_root,
    corruption_types, 
    corruption_severities=[1,2,3,4,5], 
    img_transform=cifar10_c_default_train_transforms, 
    batch_size=128, 
    num_workers=8
):
    """
    Gets dataloader for corrupted cifar100 training data (32x32)

    Args:
        cifar100_c_data_root (str): path to data where corrupted cifar100 data is saved
        corruption_types (list[str]): list of corruption types to use
        corruption_severities (list[int]): list corruption severities 0-5 (0 means clean data)
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(cifar100_c_data_root, "train_memmap"), 
        num_samples_per_corruption=50000, # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=corruption_types, 
        corruption_severities=corruption_severities, 
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

def get_cifar100_c_val(
    cifar100_c_data_root,
    corruption_types, 
    corruption_severities=[1,2,3,4,5], 
    img_transform=cifar10_c_default_val_transforms, 
    batch_size=128, 
    num_workers=4
):
    """
    Gets dataloader for corrupted cifar100 val data (32x32)

    Args:
        cifar100_c_data_root (str): path to data where corrupted cifar100 data is saved
        corruption_types (list[str]): list of corruption types to use
        corruption_severities (list[int]): list corruption severities 0-5 (0 means clean data)
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(cifar100_c_data_root, "test_memmap"), 
        num_samples_per_corruption=10000, # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=corruption_types, 
        corruption_severities=corruption_severities, 
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

def get_tin_train(tin_data_root, img_transform=cifar10_c_default_train_transforms, batch_size=128, num_workers=8):
    """
    Gets dataloader for tinyimagenet training data (32x32)

    Args:
        tin_data_root (str): path to data where tinyimagenet data is saved
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(tin_data_root, "train_memmap"), 
        num_samples_per_corruption=100000, # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=["gaussian_noise"], # intentional: corruption severity 0 will be clean images only
        corruption_severities=[0], # corruption severity 0 will be clean images only
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

def get_tin_val(tin_data_root, img_transform=cifar10_c_default_val_transforms, batch_size=128, num_workers=4):
    """
    Gets dataloader for clean tinyimagenet validation data (32x32)

    Args:
        tin_data_root (str): path to data where tinyimagenet data is saved
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(tin_data_root, "test_memmap"), 
        num_samples_per_corruption=10000, # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=["gaussian_noise"], # intentional: corruption severity 0 will be clean images only
        corruption_severities=[0], # corruption severity 0 will be clean images only
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

def get_tin_c_train(
    tin_c_data_root,
    corruption_types, 
    corruption_severities=[1,2,3,4,5], 
    img_transform=cifar10_c_default_train_transforms, 
    batch_size=128, 
    num_workers=8
):
    """
    Gets dataloader for corrupted tinyimagenet training data (32x32)

    Args:
        tin_c_data_root (str): path to data where corrupted tinyimagenet data is saved
        corruption_types (list[str]): list of corruption types to use
        corruption_severities (list[int]): list corruption severities 0-5 (0 means clean data)
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(tin_c_data_root, "train_memmap"), 
        num_samples_per_corruption=600000, # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=corruption_types, 
        corruption_severities=corruption_severities, 
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

def get_tin_c_val(
    tin_c_data_root,
    corruption_types, 
    corruption_severities=[1,2,3,4,5], 
    img_transform=cifar10_c_default_val_transforms, 
    batch_size=128, 
    num_workers=4
):
    """
    Gets dataloader for corrupted tinyimagenet validation data (32x32)

    Args:
        tin_c_data_root (str): path to data where corrupted tinyimagenet data is saved
        corruption_types (list[str]): list of corruption types to use
        corruption_severities (list[int]): list corruption severities 0-5 (0 means clean data)
        img_transform (torvision.transform): img transform
        batch_size (int): dataloader batch size
        num_workers (int): data loader num workers

    Returns:
        DataLoader
    """
    dataset = CorruptedImages(
        os.path.join(tin_c_data_root, "test_memmap"), 
        num_samples_per_corruption=60000, # total number of samples in dataset for each corruption. gets filtered down in class instantiation
        corruption_types=corruption_types, 
        corruption_severities=corruption_severities, 
        img_transform=img_transform,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

def get_train_val_data(
    data_name,
    data_root,
    batch_size=128,
    num_workers=1,
):
    """
    Gets training and test dataloaders

    Args:
        data_name (str): name of dataset
        data_root (str): path to directory containing saved data
        batch_size (int): batch size
        num_workers (int): dataloader number of workers
    
    Returns:
        train_data (DataLoader): train dataloader
        val_data (DataLoader): val dataloader
    """
    if data_name == "cifar10":
        train_data = get_cifar10_train(
            cifar10_data_root=data_root, batch_size=batch_size, num_workers=num_workers
        )
        val_data = get_cifar10_val(
            cifar10_data_root=data_root, batch_size=batch_size, num_workers=num_workers
        )
    elif data_name == "cifar10-c":
        train_data = get_cifar10_c_train(
            cifar10_c_data_root=data_root,
            corruption_types=CorruptedImages.train_corruptions,
            corruption_severities=[0,1,2,3,4,5],
            batch_size=batch_size, 
            num_workers=num_workers,
        )
        val_data = get_cifar10_c_val(
            cifar10_c_data_root=data_root, 
            corruption_types=CorruptedImages.test_corruptions,
            corruption_severities=[0,1,2,3,4,5],
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif data_name == "cifar100":
        train_data = get_cifar100_train(
            cifar100_data_root=data_root, batch_size=batch_size, num_workers=num_workers
        )
        val_data = get_cifar100_val(
            cifar100_data_root=data_root, batch_size=batch_size, num_workers=num_workers
        )
    elif data_name == "cifar100-c":
        train_data = get_cifar100_c_train(
            cifar100_c_data_root=data_root,
            corruption_types=CorruptedImages.train_corruptions,
            corruption_severities=[0,1,2,3,4,5],
            batch_size=batch_size, 
            num_workers=num_workers,
        )
        val_data = get_cifar100_c_val(
            cifar100_c_data_root=data_root, 
            corruption_types=CorruptedImages.test_corruptions,
            corruption_severities=[0,1,2,3,4,5],
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif data_name == "tin":
        train_data = get_tin_train(
            tin_data_root=data_root, batch_size=batch_size, num_workers=num_workers
        )
        val_data = get_tin_val(
            tin_data_root=data_root, batch_size=batch_size, num_workers=num_workers
        )
    elif data_name == "tin-c":
        train_data = get_tin_c_train(
            tin_c_data_root=data_root,
            corruption_types=CorruptedImages.train_corruptions,
            corruption_severities=[0,1,2,3,4,5],
            batch_size=batch_size, 
            num_workers=num_workers,
        )
        val_data = get_tin_c_val(
            tin_c_data_root=data_root, 
            corruption_types=CorruptedImages.test_corruptions,
            corruption_severities=[0,1,2,3,4,5],
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        raise NotImplementedError(f"Loading dataset {data_name} not implemented")
    
    return train_data, val_data