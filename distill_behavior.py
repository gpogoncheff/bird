import os
import fire

import numpy as np

import torch
import torch.nn as nn

from bird.utils.data_utils import dataset_nclasses, get_train_val_data
from bird.utils.cka import LinearCKALoss
from bird.models.model_factory import get_model
from bird.models.teacher_student import TeacherStudent
from bird.experiments.transfer import DistillationModelConfig, distillation_train


def distill_behavior(
    student_dataset: str,
    student_dataset_root: str,
    teacher_dataset: str,
    student_model_name: str,
    teacher_model_name: str,
    pretrained_student_model_path: str,
    pretrained_teacher_model_path: str,
    student_alignment_layer: str,
    teacher_alignment_layer: str,
    loss_alpha: float = 0.9,
    initial_lr: float = 0.005,
    num_epochs: int  = 200,
    early_stopping_patience = np.inf,
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "cuda",
    ckpt_dir: str | None = None,
):
    """
    Trains with distillation objective.

    Args:
        student_dataset (str): name of student dataset. One of cifar10, cifar100, tin
        student_dataset_root (str): path to directory containing data
        teacher_dataset (str): name of teacher dataset. One of cifar10, cifar100, tin
        student_model_name (str): name of student model.
        teacher_model_name (str): name of teacher model.
        pretrained_student_model_path (str): path to pretrained student model checkpoint.
        pretrained_teacher_model_path (str): path to pretrained teacher model checkpoint.
        student_alignment_layer (str): name of guided distillation layer
        teacher_alignment_layer (str): name of guiding distillation layer
        loss_alpha (float): representation loss weighting
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
        student_dataset,
        student_dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Instantiate teacher student model
    student_state_dict = torch.load(pretrained_student_model_path)["model"]
    student_model = get_model(
        student_model_name, num_classes=dataset_nclasses[student_dataset], state_dict=student_state_dict
    )

    teacher_state_dict = torch.load(pretrained_teacher_model_path)["model"]
    teacher_model = get_model(
        teacher_model_name, num_classes=dataset_nclasses[teacher_dataset], state_dict=teacher_state_dict
    )

    # teacher downsamling
    teacher_projection = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )

    # student downsamling
    student_projection = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )

    teacher_student = TeacherStudent(
        teacher_model=teacher_model, 
        student_model=student_model, 
        teacher_alignment_layer=teacher_alignment_layer, 
        student_alignment_layer=student_alignment_layer,
        teacher_projection=teacher_projection, 
        student_projection=student_projection,
        freeze_teacher=True,
        freeze_teacher_bn=False,
        teacher_eval=True
    )

    # Configure distillation config
    clf_loss_fn = nn.CrossEntropyLoss()
    distillation_loss_fn = LinearCKALoss
    optimizer = torch.optim.SGD(
        teacher_student.parameters(), 
        lr=initial_lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_data))

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_cfg = DistillationModelConfig(
        teacher_student_module=teacher_student, 
        train_data=train_data, 
        val_data=val_data, 
        clf_loss_fn=clf_loss_fn,
        distillation_loss_fn=distillation_loss_fn,
        loss_alpha=loss_alpha,
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        device=device, 
        num_epochs=num_epochs, 
        early_stopping_patience=early_stopping_patience,
        ckpt_dir=ckpt_dir,
    )

    _ = distillation_train(model_cfg)
    

if __name__ == "__main__":
    fire.Fire(distill_behavior)