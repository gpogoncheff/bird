import fire

import numpy as np

import torch
import torch.nn as nn

from bird.utils.data_utils import dataset_nclasses, get_train_val_data
from bird.models.model_factory import get_model
from bird.experiments.classification import ClassificationModelConfig, clf_val_epoch


def evaluate_model(
    dataset: str,
    dataset_root: str,
    model_name: str,
    trained_model_path: str | None = None,
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "cuda",
):
    """
    Evaluates accuracy of image classification model.

    Args:
        dataset (str): name of dataset. One of cifar10, cifar10-c, 
            cifar100, cifar100-c, tin, tin-c
        dataset_root (str): path to directory containing data
        model_name (str): name of model.
        trained_model_path (str | None): path to trained teacher student model checkpoint.
        batch_size (int): batch size
        num_workers (int): number of DataLoader workers
        device (str): device for training, "cpu" or "cuda"

    Returns:
        avg_acc (float): Accuracy of model over dataset
    """
    train_data, val_data = get_train_val_data(
        dataset,
        dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    ckpt = torch.load(trained_model_path, map_location="cpu")
    if "student_model" in ckpt:
        state_dict = ckpt["student_model"]
    else:
        state_dict = ckpt["model"]

    model = get_model(model_name, num_classes=dataset_nclasses[dataset], state_dict=state_dict)
    model = model.to(device)

    clf_cfg = ClassificationModelConfig(
        model=model,
        train_data=train_data,
        val_data=val_data,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=None,
        lr_scheduler=None,
        num_epochs=-1,
        ckpt_dir=None,
        device=device,
    )

    _, avg_acc = clf_val_epoch(clf_model_cfg=clf_cfg, epoch=-1)
    print(f"Accuracy: {avg_acc:.4f}")
    return avg_acc


if __name__ == "__main__":
    fire.Fire(evaluate_model)