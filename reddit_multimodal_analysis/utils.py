import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from config import Config
import numpy as np
import random
import os


def get_scheduler(optimizer: optim):
    """
    A method which returns the required schedulers.
        - Extracted from Awsaf's Kaggle.
    """
    if Config.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=Config.T_max, eta_min=Config.min_lr
        )
    elif Config.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=Config.T_0, eta_min=Config.eta_min
        )
    elif Config.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            min_lr=Config.min_lr,
        )
    elif Config.scheduler == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.85)
    elif Config.scheduler is None:
        scheduler = None
    else:
        raise NotImplementedError(
            "The Scheduler you have asked has not been implemented"
        )
    return scheduler


def get_optimizer(model: nn.Module):
    """
    Returns the optimizer based on the Config files.
    """
    if Config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(
            model.parameters(), lr=Config.learning_rate, rho=Config.rho, eps=Config.eps
        )
    elif Config.optimizer == "Adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=Config.learning_rate,
            lr_decay=Config.lr_decay,
            weight_decay=Config.weight_decay,
        )
    elif Config.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.learning_rate,
            betas=Config.betas,
            eps=Config.eps,
        )
    elif Config.optimizer == "RMSProp":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=Config.learning_rate,
            alpha=Config.alpha,
            eps=Config.eps,
            weight_decay=Config.weight_decay,
            momentum=Config.momentum,
        )
    else:
        raise NotImplementedError(
            f"The optimizer {Config.optimizer} has not been implemented."
        )
    return optimizer


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(">>> SEEDED <<<")
