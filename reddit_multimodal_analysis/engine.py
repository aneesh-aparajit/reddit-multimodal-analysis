import copy
import gc
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from colorama import Back, Fore, Style
from config import Config
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from tqdm import tqdm

c_ = Fore.GREEN
sr_ = Style.RESET_ALL


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
    f1_metrics = F1Score(task="multiclass", num_classes=Config.num_classes)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(train) ")
    for step, batch in pbar:
        batch = {k: v.to(Config.device) for k, v in batch.items()}
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
        f1 = f1_metrics(yHat, labels)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "train/loss": epoch_loss,
                "train/accuracy": accuracy,
                "train/precision": precision,
                "train/recall": recall,
                "train/auroc": auroc,
                "train/f1": f1,
                "train/current_lr": current_lr,
            },
            step=step,
        )

        pbar.set_postfix(epoch_loss=f"{epoch_loss:.5f}", current_lr=f"{current_lr:.5f}")

    return epoch_loss


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, dataloader: DataLoader
) -> Tuple[float, defaultdict(list)]:
    model.train()
    dataset_size = 0
    running_loss = 0

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=Config.num_classes)
    precision_metric = Precision(task="multiclass", num_classes=Config.num_classes)
    recall_metric = Recall(task="multiclass", num_classes=Config.num_classes)
    auroc_metric = AUROC(task="multiclass", num_classes=Config.num_classes)
    f1_metrics = F1Score(task="multiclass", num_classes=Config.num_classes)

    val_scores = defaultdict(list)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(valid) ")
    for step, batch in pbar:
        batch = {k: v.to(Config.device) for k, v in batch.items()}
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
        f1 = f1_metrics(yHat, labels)

        val_scores["accuracy"].append(accuracy)
        val_scores["precision"].append(precision)
        val_scores["recall"].append(recall)
        val_scores["auroc"].append(auroc)
        val_scores["f1"].append(f1)

        wandb.log(
            {
                "valid/loss": epoch_loss,
                "valid/accuracy": accuracy,
                "valid/precision": precision,
                "valid/recall": recall,
                "valid/auroc": auroc,
                "valid/f1": f1,
            },
            step=step,
        )

    return epoch_loss, val_scores


def run_training(
    model: nn.Module,
    optimizer: optim,
    trainloader: DataLoader,
    validloader: DataLoader,
    run: wandb,
    fold: int,
    scheduler: lr_scheduler = None,
) -> Tuple[nn.Module, defaultdict]:
    wandb.watch(models=[model], log_freq=100)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(Config.epochs):
        gc.collect()
        print(f"\t\t\t\t########## EPOCH [{epoch+1}/{Config.epochs}] ##########")
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=trainloader,
        )
        valid_loss, valid_scores = validate_one_epoch(
            model=model, dataloader=validloader
        )

        wandb.log(
            {
                "train/epoch/loss": train_loss,
                "valid/epoch/loss": valid_loss,
                "valid/epoch/accuracy": np.mean(valid_scores["accuracy"]),
                "valid/epoch/precision": np.mean(valid_scores["precision"]),
                "valid/epoch/recall": np.mean(valid_scores["recall"]),
                "valid/epoch/auroc": np.mean(valid_scores["auroc"]),
                "valid/epoch/f1": np.mean(valid_scores["f1"]),
                "current_lr": optimizer.param_groups[0]["lr"],
            }
        )

        history["accuracy"].append(np.mean(valid_scores["accuracy"]))
        history["precision"].append(np.mean(valid_scores["precision"]))
        history["recall"].append(np.mean(valid_scores["recall"]))
        history["auroc"].append(np.mean(valid_scores["auroc"]))
        history["f1"].append(np.mean(valid_scores["f1"]))

        print(
            f'Valid Accuracy: {np.mean(valid_scores["accuracy"]):.5f} | Valid Loss: {valid_loss:.5f}'
        )

        if valid_loss < best_loss:
            print(
                f"{c_}Validation Score Improved from {best_loss:.5f} to {valid_loss:.5f}"
            )
            best_epoch = epoch + 1
            best_loss = valid_loss
            run.summary["Best Loss"] = best_loss
            run.summary["Best Epoch"] = best_epoch
            run.summary["Best Accuracy"] = np.mean(valid_scores["accuracy"])
            run.summary["Best Precision"] = np.mean(valid_scores["precision"])
            run.summary["Best Recall"] = np.mean(valid_scores["recall"])
            run.summary["Best AUROC"] = np.mean(valid_scores["auroc"])
            run.summary["Best F1 Score"] = np.mean(valid_scores["f1"])

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"../artifacts/models/best_epoch-{fold:02d}.bin"
            torch.save(obj=best_model_wts, f=PATH)
            wandb.save(PATH)
            print(f"MODEL SAVED!{sr_}")

        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"../artifacts/models/last_epoch-{fold:02d}.bin"
        torch.save(last_model_wts, PATH)

    model.load_state_dict(best_model_wts, strict=True)
    torch.save(history, f=f"../artifacts/history/fold-{fold:02d}.pth")
    return model, history
