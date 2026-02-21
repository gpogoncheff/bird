import os
import fire

import numpy as np

import torch
import torch.nn as nn

from bird.utils.data_utils import dataset_nclasses, get_train_val_data
from bird.models.model_factory import get_model
from bird.experiments.classification import ClassificationModelConfig, clf_train


def pretrain_classification(
    dataset: str,
    dataset_root: str,
    model_name: str,
    pretrained_model_path: str | None = None,
    initial_lr: float = 0.01,
    num_epochs: int  = 200,
    early_stopping_patience = np.inf,
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "cuda",
    ckpt_dir: str | None = None,
):
    """
    Trains a model for image classification.

    Args:
        dataset (str): name of dataset. One of cifar10, cifar10-c, 
            cifar100, cifar100-c, tin, tin-c
        dataset_root (str): path to directory containing data
        model_name (str): name of model.
        pretrained_model_path (str | None): path to pretrained model checkpoint.
            None if training from scratch.
        initial_lr (float): initial learning rate
        num_epochs (int): number of epochs
        early_stopping_patience: Stop afer plateau on val data for this many epochs
        batch_size (int): batch size
        num_workers (int): number of DataLoader workers
        device (str): device for training, "cpu" or "cuda"
        ckpt_dir (str): path to where checkpoints should be saved

    Returns:
        None
    """
    train_data, val_data = get_train_val_data(
        dataset,
        dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path)["model"]
    else:
        state_dict = None
    model = get_model(model_name, num_classes=dataset_nclasses[dataset], state_dict=state_dict)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=initial_lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs*len(train_data), 
        eta_min=1e-6
    )

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    clf_cfg = ClassificationModelConfig(
        model=model,
        train_data=train_data,
        val_data=val_data,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        ckpt_dir=ckpt_dir,
        device=device,
    )

    _ = clf_train(clf_model_cfg=clf_cfg)
    return


if __name__ == "__main__":
    fire.Fire(pretrain_classification)