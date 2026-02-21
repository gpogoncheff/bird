from copy import deepcopy
import numpy as np
import os
import torch
from tqdm import tqdm


class DistillationModelConfig():
    """
    Configuration for knowledge distillation task.

    Parameters:
        teacher_student_module (TeacherStudent): teacher student module
        train_data (DataLoader): training data
        val_data (DataLoader): validation data
        clf_loss_fn (fn): classifcation loss function
        distillation_loss_fn (fn): distillation loss function
        loss_alpha (float): weight [0,1] balancing classifciation and distillation loss
        optimizer (Optimizer): torch optimizer
        lr_scheduler (lr_scheduler): learning rate scheduler
        device (str): training/inference device type
        num_epochs (int): number of training epochs
        early_stopping_patience (int): stop training after plateau for this many epochs
        ckpt_dir (str): directory at which model checkpoints will be saved
    """
    def __init__(
        self, 
        teacher_student_module, 
        train_data, 
        val_data, 
        clf_loss_fn,
        distillation_loss_fn,
        loss_alpha,
        optimizer, 
        lr_scheduler, 
        device, 
        num_epochs, 
        early_stopping_patience=np.inf,
        ckpt_dir=None,
    ):
        self.teacher_student_module = teacher_student_module
        self.train_data = train_data
        self.val_data = val_data
        self.clf_loss_fn = clf_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.loss_alpha = loss_alpha
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.ckpt_dir = ckpt_dir


def distillation_val_epoch(model_cfg, epoch=-1):
    """
    Performs inference over validation set.

    Args:
        model_cfg (DistillationModelConfig): distillation model configuration
        epoch (int): current epoch

    Returns
        avg_clf_loss (float): classification loss averaged over validation data
        avg_distill_loss (float): distillationloss averaged over validation data
        avg_loss (float): total loss averaged over validation data
        avg_acc (float): accuracy averaged over validation data
    """
    model_cfg.teacher_student_module.set_is_eval()

    n_samples, running_clf_loss, running_distill_loss, running_total_loss, running_correct = 0, 0, 0, 0, 0
    val_progress = tqdm(model_cfg.val_data, desc=f"Val epoch {epoch}  ")
    with torch.no_grad():
        for (img, label) in val_progress:
            img = img.to(model_cfg.device)
            label = label.to(model_cfg.device)

            (t_out, s_out), (t_f, s_f) = model_cfg.teacher_student_module(img)
            clf_loss = model_cfg.clf_loss_fn(s_out, label)
            distill_loss = model_cfg.distillation_loss_fn(s_f, t_f)
            total_loss = (
                ((1-model_cfg.loss_alpha)*clf_loss) + 
                (model_cfg.loss_alpha*distill_loss)
            )

            num_correct = torch.sum(torch.argmax(s_out, dim=1) == label).item()
            running_clf_loss += clf_loss.item()*len(label)
            running_distill_loss += distill_loss.item()*len(label)
            running_total_loss += total_loss.item()*len(label)
            running_correct += num_correct
            n_samples += len(label)

            avg_clf_loss = running_clf_loss/n_samples
            avg_distill_loss = running_distill_loss/n_samples
            avg_total_loss = running_total_loss/n_samples
            avg_acc = running_correct/n_samples
            val_progress.set_postfix({
                "clf_loss": f"{avg_clf_loss:.4f}", 
                "distill_loss": f"{avg_distill_loss:.4f}", 
                "total_loss": f"{avg_total_loss:.4f}", 
                "acc": f"{avg_acc:.4f}"
            })

    return avg_clf_loss, avg_distill_loss, avg_total_loss, avg_acc
    

def distillation_train_epoch(model_cfg, epoch=-1):
    """
    Performs one epoch of training.

    Args:
        model_cfg (DistillationModelConfig): distillation model configuration
        epoch (int): current epoch

    Returns
        avg_clf_loss (float): classification loss averaged over training data
        avg_distill_loss (float): distillationloss averaged over training data
        avg_loss (float): total loss averaged over training data
        avg_acc (float): accuracy averaged over training data
    """
    model_cfg.teacher_student_module.set_is_training()
    n_samples, running_clf_loss, running_distill_loss, running_total_loss, running_correct = 0, 0, 0, 0, 0
    train_progress = tqdm(model_cfg.train_data, desc=f"Train epoch {epoch}")
    for (img, label) in train_progress:
        model_cfg.teacher_student_module.zero_grad()
        img = img.to(model_cfg.device)
        label = label.to(model_cfg.device)
        
        (t_out, s_out), (t_f, s_f) = model_cfg.teacher_student_module(img)
        clf_loss = model_cfg.clf_loss_fn(s_out, label)
        distill_loss = model_cfg.distillation_loss_fn(s_f, t_f)
        total_loss = (
            ((1-model_cfg.loss_alpha)*clf_loss) + 
            (model_cfg.loss_alpha*distill_loss)
        )
        total_loss.backward()
        model_cfg.optimizer.step()
        model_cfg.lr_scheduler.step()

        num_correct = torch.sum(torch.argmax(s_out, dim=1) == label).item()
        running_clf_loss += clf_loss.item()*len(label)
        running_distill_loss += distill_loss.item()*len(label)
        running_total_loss += total_loss.item()*len(label)
        running_correct += num_correct
        n_samples += len(label)

        avg_clf_loss = running_clf_loss/n_samples
        avg_distill_loss = running_distill_loss/n_samples
        avg_total_loss = running_total_loss/n_samples
        avg_acc = running_correct/n_samples
        train_progress.set_postfix({
            "clf_loss": f"{avg_clf_loss:.4f}", 
            "distill_loss": f"{avg_distill_loss:.4f}", 
            "total_loss": f"{avg_total_loss:.4f}", 
            "acc": f"{avg_acc:.4f}"
        })
    
    return avg_clf_loss, avg_distill_loss, avg_total_loss, avg_acc


def distillation_train(model_cfg):
    """
    Trains model according to specifications in model_cfg

    Args:
        model_cfg (DistillationModelConfig): distillation model configuration
    Returns:
        hist (dict): summary of training history
    """
    hist = {
        "train_acc": [],
        "train_clf_loss": [],
        "train_distillation_loss": [],
        "train_total_loss": [],
        "val_acc": [],
        "val_clf_loss": [],
        "val_distillation_loss": [],
        "val_total_loss": [],
        "optimizer": model_cfg.optimizer.state_dict(),
        "lr_scheduler": model_cfg.lr_scheduler.state_dict(),
        "model": model_cfg.teacher_student_module.state_dict(),
    }
    model_cfg.teacher_student_module.to(model_cfg.device)
    early_stopping_patience = model_cfg.early_stopping_patience
    print("Performance, pre-fine-tuning:")
    original_results = distillation_val_epoch(model_cfg, epoch=-1)
    hist["original_val_acc"] = original_results[-1]
    print("Training...")
    min_total_loss = np.inf
    for epoch in range(model_cfg.num_epochs):
        train_clf_loss, train_distillation_loss, train_total_loss, train_acc = distillation_train_epoch(model_cfg, epoch)
        val_clf_loss, val_distillation_loss, val_total_loss, val_acc = distillation_val_epoch(model_cfg, epoch)
        
        hist["train_clf_loss"].append(train_clf_loss)
        hist["train_distillation_loss"].append(train_distillation_loss)
        hist["train_total_loss"].append(train_total_loss)
        hist["train_acc"].append(train_acc)
        hist["val_clf_loss"].append(val_clf_loss)
        hist["val_distillation_loss"].append(val_distillation_loss)
        hist["val_total_loss"].append(val_total_loss)
        hist["val_acc"].append(val_acc)
        hist["model"] = model_cfg.teacher_student_module.state_dict()
        hist["teacher_model"] = model_cfg.teacher_student_module.teacher_model.state_dict()
        hist["student_model"] = model_cfg.teacher_student_module.student_model.state_dict()
        hist["optimizer"] = model_cfg.optimizer.state_dict()
        hist["lr_scheduler"] = model_cfg.lr_scheduler.state_dict()

        if (val_total_loss < min_total_loss):
            min_total_loss = val_total_loss
            early_stopping_patience = model_cfg.early_stopping_patience
            if (model_cfg.ckpt_dir is not None):
                ckpt_path = os.path.join(model_cfg.ckpt_dir, "best_loss.pt")
                torch.save(hist, ckpt_path)
        else:
            early_stopping_patience -= 1
            if (early_stopping_patience <= 0):
                print("Early stopping: val acc plateaued")
                break

    return hist