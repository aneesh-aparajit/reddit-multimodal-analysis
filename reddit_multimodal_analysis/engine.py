from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from config import Config
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    optimizer: optim,
    dataloader: DataLoader,
    scheduler=None,
) -> float:
    model.train()
    dataset_size = 0
    running_loss = 0

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=Config.num_classes)
    precision_metric = Precision(task="multiclass", num_classes=Config.num_classes)
    recall_metric = Recall(task="multiclass", num_classes=Config.num_classes)
    auroc_metric = AUROC(task="multiclass", num_classes=Config.num_classes)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(train) ")
    for step, batch in pbar:
        labels = batch["label"]
        yHat = model.forward(**batch)

        optimizer.zero_grad()
        loss = criterion(yHat, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * labels.shape[0]
        dataset_size += labels.shape[0]

        epoch_loss = running_loss / dataset_size

        out = torch.argmax(yHat, axis=1)
        accuracy = accuracy_metric(out, labels)
        precision = precision_metric(out, labels)
        recall = recall_metric(out, labels)
        auroc = auroc_metric(yHat, labels)

        wandb.log(
            {
                "train/loss": epoch_loss,
                "train/accuracy": accuracy,
                "train/precision": precision,
                "train/recall": recall,
                "train/auroc": auroc,
            },
            step=step,
        )

    return epoch_loss


@torch.no_grad()
def validate_one_epoch(model: nn.Module, dataloader: DataLoader) -> float:
    model.train()
    dataset_size = 0
    running_loss = 0

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=Config.num_classes)
    precision_metric = Precision(task="multiclass", num_classes=Config.num_classes)
    recall_metric = Recall(task="multiclass", num_classes=Config.num_classes)
    auroc_metric = AUROC(task="multiclass", num_classes=Config.num_classes)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(valid) ")
    for step, batch in pbar:
        labels = batch["label"]
        yHat = model.forward(**batch)

        loss = criterion(yHat, labels)

        running_loss += loss.item() * labels.shape[0]
        dataset_size += labels.shape[0]

        epoch_loss = running_loss / dataset_size

        out = torch.argmax(yHat, axis=1)
        accuracy = accuracy_metric(out, labels)
        precision = precision_metric(out, labels)
        recall = recall_metric(out, labels)
        auroc = auroc_metric(yHat, labels)

        wandb.log(
            {
                "valid/loss": epoch_loss,
                "valid/accuracy": accuracy,
                "valid/precision": precision,
                "valid/recall": recall,
                "valid/auroc": auroc,
            },
            step=step,
        )

    return epoch_loss
