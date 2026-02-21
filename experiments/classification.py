import os
import numpy as np
import torch
from tqdm import tqdm


class ClassificationModelConfig():
    """
    Configuration for vanilla classification task.

    Params:
        model (nn.Module): torch model
        train_data (DataLoader): dataloader for training data
        val_data (DataLoader): dataloader for validation data
        loss_fn (fn): classifcation loss function
        optimizer (Optimizer): torch optimizer
        lr_scheduler (lr_scheduler): learning rate scheduler
        device (str): training/inference device type
        num_epochs (int): number of training epochs
        early_stopping_patience (int): stop training after plateau for this many epochs
        ckpt_dir (str): directory at which model checkpoints will be saved
    """
    def __init__(
        self, 
        model, 
        train_data, 
        val_data, 
        loss_fn, 
        optimizer, 
        lr_scheduler, 
        device, 
        num_epochs, 
        early_stopping_patience=np.inf,
        ckpt_dir=None,
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.ckpt_dir = ckpt_dir


def clf_val_epoch(clf_model_cfg, epoch=-1):
    """
    Performs inference over validation set.

    Args:
        clf_model_cfg (ClassificationModelConfig): classification model configuration
        epoch (int): current epoch

    Returns
        avg_loss (float): loss averaged over validation set
        avg_acc (float): accuracy averaged over validation data
    """
    clf_model_cfg.model.eval()
    n_samples, running_loss, running_correct = 0, 0, 0
    val_progress = tqdm(clf_model_cfg.val_data, desc=f"Val epoch {epoch}  ")
    with torch.no_grad():
        for (img, label) in val_progress:
            img = img.to(clf_model_cfg.device)
            label = label.to(clf_model_cfg.device)

            model_output = clf_model_cfg.model(img)
            if isinstance(model_output, tuple) and (len(model_output) == 2):
                model_output = model_output[1]
            loss = clf_model_cfg.loss_fn(model_output, label)
            num_correct = torch.sum(torch.argmax(model_output, dim=1) == label).item()

            running_loss += loss.item()*len(label)
            running_correct += num_correct
            n_samples += len(label)
            avg_loss = running_loss/n_samples
            avg_acc = running_correct/n_samples
            val_progress.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})

    return avg_loss, avg_acc


def clf_train_epoch(clf_model_cfg, epoch=-1):
    """
    Performs one epoch of training.

    Args:
        clf_model_cfg (ClassificationModelConfig): classification model configuration
        epoch (int): current epoch

    Returns
        avg_loss (float): loss averaged over training set
        avg_acc (float): accuracy averaged over training data
    """
    clf_model_cfg.model.train()
    n_samples, running_loss, running_correct = 0, 0, 0
    train_progress = tqdm(clf_model_cfg.train_data, desc=f"Train epoch {epoch}")
    for (img, label) in train_progress:
        clf_model_cfg.model.zero_grad()
        img = img.to(clf_model_cfg.device)
        label = label.to(clf_model_cfg.device)
        model_output = clf_model_cfg.model(img)
        loss = clf_model_cfg.loss_fn(model_output, label)
        loss.backward()
        clf_model_cfg.optimizer.step()
        clf_model_cfg.lr_scheduler.step()

        num_correct = torch.sum(torch.argmax(model_output, dim=1) == label).item()
        running_loss += loss.item()*len(label)
        running_correct += num_correct
        n_samples += len(label)
        avg_loss = running_loss/n_samples
        avg_acc = running_correct/n_samples
        train_progress.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})
    
    return avg_loss, avg_acc


def clf_train(clf_model_cfg):
    """
    Trains model according to specifications in clf_model_cfg

    Args:
        clf_model_cfg (ClassificationModelConfig): classification model configuration
    Returns:
        hist (dict): summary of training history
    """
    hist = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
        "optimizer": clf_model_cfg.optimizer.state_dict(),
        "lr_scheduler": clf_model_cfg.lr_scheduler.state_dict(),
        "model": clf_model_cfg.model.state_dict(),
    }
    min_loss, max_acc = np.inf, 0
    clf_model_cfg.model.to(clf_model_cfg.device)
    early_stopping_patience = clf_model_cfg.early_stopping_patience
    for epoch in range(clf_model_cfg.num_epochs):
        train_loss, train_acc = clf_train_epoch(clf_model_cfg, epoch)
        val_loss, val_acc = clf_val_epoch(clf_model_cfg, epoch)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["model"] = clf_model_cfg.model.state_dict()
        hist["optimizer"] = clf_model_cfg.optimizer.state_dict()
        hist["lr_scheduler"] = clf_model_cfg.lr_scheduler.state_dict()

        if (val_acc > max_acc):
            max_acc = val_acc
            if (clf_model_cfg.ckpt_dir is not None):
                ckpt_path = os.path.join(clf_model_cfg.ckpt_dir, "best_acc.pt")
                torch.save(hist, ckpt_path)

        if (val_loss < min_loss):
            min_loss = val_loss
            early_stopping_patience = clf_model_cfg.early_stopping_patience
            if (clf_model_cfg.ckpt_dir is not None):
                ckpt_path = os.path.join(clf_model_cfg.ckpt_dir, "best_loss.pt")
                torch.save(hist, ckpt_path)
        else:
            early_stopping_patience -= 1
            if (early_stopping_patience <= 0):
                print("Early stopping: val loss plateaued")
                break

    return hist
