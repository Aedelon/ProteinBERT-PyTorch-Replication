#!/usr/bin/env python
# *** coding: utf-8 ***

"""utils.py: Contains helper functions to create dataloaders. In addiction, this module contains pretraining
function, training function and testing function.

* Author: Delanoe PIRARD
* Email: delanoe.pirard.pro@gmail.com
* Licence: MIT
"""

# IMPORTS -------------------------------------------------
import os
import torch.utils.data
import logging
import data_processing

from timeit import default_timer as time
from PIL import Image
from typing import List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, SequentialLR, LambdaLR, CosineAnnealingLR
from tqdm.auto import tqdm


# FUNCTIONS -----------------------------------------------
def optimal_num_workers_testing(dataset: torch.utils.data.Dataset) -> None:
    """Take a dataset and print the elapsed time for 2 epochs by num_workers.

    :param dataset: A dataset.
    :return: /
    """
    train_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=0,
        batch_size=64,
        pin_memory=True
    )

    start = time()
    for epoch in range(2):
        for _, _ in enumerate(train_loader, 0):
            pass
    end = time()
    print(f"Finish with:{(end - start)} second, num_workers=0")

    train_loader = DataLoader(dataset=dataset, shuffle=True, num_workers=1, batch_size=64, pin_memory=True)

    start = time()
    for epoch in range(2):
        for _, _ in enumerate(train_loader, 0):
            pass
    end = time()
    print(f"Finish with:{(end - start)} second, num_workers=1")

    for num_workers in range(2, os.cpu_count(), 2):
        train_loader = DataLoader(dataset=dataset, shuffle=True, num_workers=1, batch_size=64, pin_memory=True)

        start = time()
        for epoch in range(2):
            for _, _ in enumerate(train_loader, 0):
                pass
        end = time()
        print(f"Finish with:{(end - start)} second, num_workers={num_workers}")


def create_pretrain_dataloaders(train_dir: str,
                                batch_size: int,
                                recursive_dir: bool = False,
                                num_workers: int = 0) -> Tuple:
    """Creates training and testing DataLoaders.

    Takes in a training image directory and testing image directory path, turns them into PyTorch Datasets and then
    PyTorch DataLoaders.

    :param train_dir: Path to training directory.
    :param batch_size: Number of samples per batch in each of the DataLoaders.
    :param recursive_dir: If the function checks in the train_dir recursively.
    :param num_workers: An integer for number of workers per DataLoader.
    :return: A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(
                train_dir=path/to/train_dir,
                test_dir=path/to/test_dir,
                transform=some_transform,
                batch_size=32,
                num_workers=4
            )
    """
    # Use ImageFolder to create datasets
    train_data = data_processing.UniRefGO_HDF5PretrainingDataset(train_dir, recursive=recursive_dir, load_data=False)

    # Turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               metrics: Dict[str, Any],
               gradient_clipping: bool = True,
               gradient_clipping_thresh: float = 1,
               device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')) \
        -> Tuple[float, Dict[str, float]]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    :param gradient_clipping_thresh:
    :param gradient_clipping:
    :param model: A PyTorch model to be trained.
    :param dataloader: A DataLoader instance for the model to be trained on.
    :param loss_fn: A PyTorch loss function to minimize.
    :param optimizer: A PyTorch optimizer to help minimize the loss function.
    :param metrics: A Dictionary containing metrics associated with their string names.
    :param device: The device type Pytorch will use ('cuda', 'mps' or 'cpu').
    :return: A tuple of training loss and training accuracy metrics.
    In the form (train_loss: float, train_metrics: Dict[str, float]).

    For example: (0.1112, {"Accuracy": 0.76, "AUROC": 0.56})
    """
    model.train()

    train_loss = 0
    train_metrics = {}
    for key, _ in metrics.items():
        train_metrics[key] = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred_logits = model(X)

        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_thresh)

        optimizer.step()

        y_preds = torch.softmax(y_pred_logits, dim=1)
        for key, metric in metrics.items():
            train_metrics[key] += float(metric(y_preds, y).detach().numpy())

    train_loss /= len(dataloader)
    for key, value in train_metrics.items():
        train_metrics[key] /= len(dataloader)

    return train_loss, train_metrics


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              metrics: Dict[str, Any],
              device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')) \
        -> Tuple[float, Dict[str, float]]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset.

    :param model: A PyTorch model to be trained.
    :param dataloader: A DataLoader instance for the model to be trained on.
    :param loss_fn: A PyTorch loss function to minimize.
    :param metrics: A Dictionary containing metrics associated with their string names.
    :param device: The device type Pytorch will use ('cuda', 'mps' or 'cpu').
    :return: A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss: float, test_metrics: Dict[str, float]). `

    For example: (0.1112, {"Accuracy": 0.76, "AUROC": 0.56})
    """
    model.eval()

    test_loss = 0
    test_metrics = {}
    for key, _ in metrics.items():
        test_metrics[key] = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)

            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            y_preds = torch.softmax(y_pred_logits, dim=1)
            for key, metric in metrics.items():
                test_metrics[key] += float(metric(y_preds, y).detach().numpy())

        test_loss /= len(dataloader)
        for key, value in test_metrics.items():
            test_metrics[key] /= len(dataloader)

    return test_loss, test_metrics


def pretrain(model: torch.nn.Module,
             train_dataloader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Optimizer,
             local_loss_fn: torch.nn.Module,
             global_loss_fn: torch.nn.Module,
             max_batch_iterations: int,
             save_path: str,
             nb_iterations_checkpoint: int = 1000,
             optim_scheduler_patience: int = 25,
             warmup_duration: int = 10000,
             loaded_checkpoint: Dict[str, Any] = None,
             device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, Any]:
    """Pretrain a PyTorch model.

    Calculates, prints and stores evaluation metrics throughout.

    :param model: A PyTorch model to be trained and tested.
    :param train_dataloader: A DataLoader instance for the model to be trained on.
    :param optimizer: A PyTorch optimizer to help minimize the loss function.
    :param local_loss_fn: A PyTorch loss function to calculate loss on local part of the dataset.
    :param global_loss_fn: A PyTorch loss function to calculate loss on global part of the datasets.
    :param max_batch_iterations: Maximum number of iterations for the pretraining.
    :param save_path: Path where to save model or checkpoints.
    :param nb_iterations_checkpoint: Number of iterations between each checkpoints.
    :param optim_scheduler_patience: Number of iterations where the loss isn't evolving before
    lowering the learning rate.
    :param warmup_duration: Number of iteration for pretraining warming up.
    :param loaded_checkpoint: A Dictionary containing the value to resume the pretraining.
    :param device: The device type Pytorch will use ('cuda', 'mps' or 'cpu').
    :return: A dictionary of pretraining loss as well as training configured metrics.
    Each metric has a value in a list for each iteration.
    """
    results = {
        "train_loss": []
    }

    # Variable initialization
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=optim_scheduler_patience)

    def linear_warmup(current_step: int):
        return float(current_step / warmup_duration)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)

    full_scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], [warmup_duration])

    # Load checkpoint
    if loaded_checkpoint is not None:
        logging.info("Loading checkpoint...")
        current_batch_iteration = loaded_checkpoint['current_batch_iteration']
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(loaded_checkpoint["scheduler_state_dict"])
        warmup_scheduler.load_state_dict(loaded_checkpoint["warmup_scheduler_state_dict"])
        full_scheduler.load_state_dict(loaded_checkpoint["full_scheduler_state_dict"])
        logging.info("Checkpoint loaded!")
    else:
        current_batch_iteration = 0

    model.train()

    # Pretraining iterations
    while current_batch_iteration < max_batch_iterations:
        for batch, (X, Y, sample_weights) in enumerate(train_dataloader):
            start_time = time()
            train_loss = 0

            X["local"], X["global"], Y["local"], Y["global"], sample_weights["local"], sample_weights["global"] = \
                X["local"].to(device), X["global"].to(device), Y["local"].to(device), Y["global"].to(device), \
                sample_weights["local"].to(device), sample_weights["global"].to(device)

            local_Y_pred_logits, global_Y_pred_logits = model(X)

            loss = torch.mean(local_loss_fn(local_Y_pred_logits.permute(0, 2, 1), Y["local"]) * sample_weights["local"]) \
                + torch.mean(global_loss_fn(global_Y_pred_logits, Y["global"].float()) * sample_weights["global"])
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            #local_Y_preds = torch.softmax(local_Y_pred_logits, dim=1)
            #global_Y_preds = torch.sigmoid(global_Y_pred_logits, dim=1)

            end_time = time()
            # Print metrics by epoch
            logging.info(
                f"Current batch iteration: {current_batch_iteration + 1} | "
                f"Train loss: {train_loss:.4f} | "
                f"Learning rate: {full_scheduler.get_last_lr()[0]} | "
                f"Batch Iteration time: {end_time - start_time:.4f} seconds"
            )

            results["train_loss"].append(train_loss)

            current_batch_iteration += 1

            full_scheduler.step()

            if current_batch_iteration >= max_batch_iterations:
                break

            if current_batch_iteration % nb_iterations_checkpoint == 0:
                # Save checkpoint
                logging.info(f"Saving checkpoint to {save_path}...")
                torch.save({
                    "current_batch_iteration": current_batch_iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
                    "full_scheduler_state_dict": full_scheduler.state_dict(),
                    "loss": results["train_loss"][-1],
                }, Path(save_path) / f"proteinbert_pretraining_checkpoint_{current_batch_iteration}.pt")
                logging.info(f"Checkpoint saved to {save_path} as "
                             f"protein_pretraining_checkpoint_{current_batch_iteration}.pt")

    # Save the whole model
    logging.info(f"Save whole model to {save_path}...")
    torch.save(model, Path(save_path) / f"proteinbert_pretrained_model_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.pt")
    logging.info(f"Whole model saved to {save_path} "
                 f"as proteinbert_pretrained_model_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.pt")

    return results


# def train(model: torch.nn.Module,
#           train_dataloader: torch.utils.data.DataLoader,
#           test_dataloader: torch.utils.data.DataLoader,
#           optimizer: torch.optim.Optimizer,
#           loss_fn: torch.nn.Module,
#           metrics: Dict[str, Any],
#           epochs: int,
#           save_path: str,
#           loaded_checkpoint: Dict[str, Any] = None,
#           device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, Any]:
#     """Trains and tests a PyTorch model.
#
#     Passes a target PyTorch models through train_step() and test_step() functions for a number of epochs,
#     training and testing the model in the same epoch loop.
#
#     Calculates, prints and stores evaluation metrics throughout.
#
#     :param model: A PyTorch model to be trained and tested.
#     :param train_dataloader: A DataLoader instance for the model to be trained on.
#     :param test_dataloader: A DataLoader instance for the model to be tested on.
#     :param optimizer: A PyTorch optimizer to help minimize the loss function.
#     :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
#     :param metrics: A Dictionary containing metrics associated with their string names.
#     :param epochs: An integer indicating how many epochs to train for.
#     :param loaded_checkpoint: A Dictionary containing the value to resume the training
#     :param device: The device type Pytorch will use ('cuda', 'mps' or 'cpu').
#     :param save_path: Path where to save model or checkpoints.
#     :return: A dictionary of training and testing loss as well as training and testing configured metrics.
#     Each metric has a value in a list for each epoch.
#     """
#     results = {
#         "train_loss": [],
#         "train_metrics": {metric: [] for metric in list(metrics.keys())},
#         "validation_loss": [],
#         "validation_metrics": {metric: [] for metric in list(metrics.keys())}
#     }
#
#     # Variables initialization
#     scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
#
#     # Load checkpoint
#     if loaded_checkpoint is not None:
#         logging.info("Loading checkpoint...")
#         start_epoch = loaded_checkpoint['epoch']
#         model.load_state_dict(loaded_checkpoint["model_state_dict"])
#         optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
#         scheduler.load_state_dict(loaded_checkpoint["scheduler_state_dict"])
#         logging.info("Checkpoint loaded!")
#     else:
#         start_epoch = 0
#
#     # Training iterations
#     for epoch in tqdm(range(start_epoch, epochs, 1)):
#         start_time = time()
#         train_loss, train_metrics = train_step(
#             model=model,
#             dataloader=train_dataloader,
#             loss_fn=loss_fn,
#             optimizer=optimizer,
#             metrics=metrics,
#             device=device
#         )
#         val_loss, val_metrics = test_step(
#             model=model,
#             dataloader=test_dataloader,
#             loss_fn=loss_fn,
#             metrics=metrics,
#             device=device
#         )
#         scheduler.step()
#
#         end_time = time()
#
#         # Print metrics by epoch
#         print(
#             f"Epoch: {epoch + 1} | "
#             f"train_loss: {train_loss:.4f} | "
#             f"train_metrics: {train_metrics} | "
#             f"validation_loss: {val_loss:.4f} | "
#             f"validation_metrics: {val_metrics} | "
#             f"Learning rate: {scheduler.get_last_lr()[0]} | "
#             f"Epoch time: {end_time - start_time:.4f} seconds"
#         )
#
#         results["train_loss"].append(train_loss)
#
#         for key, metric in results["train_metrics"].items():
#             results["train_metrics"][key].append(train_metrics[key])
#
#         results["validation_loss"].append(val_loss)
#
#         for key, metric in results["validation_metrics"].items():
#             results["validation_metrics"][key].append(val_metrics[key])
#
#         # Save checkpoint at each epoch
#         logging.info(f"Saving checkpoint to {save_path}...")
#         torch.save({
#             "epoch": epoch,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "scheduler_state_dict": scheduler.state_dict(),
#             "loss": val_loss
#         }, Path(save_path) / f"vit_checkpoint_epoch_{epoch}.pt")
#         logging.info(f"Checkpoint saved to {save_path} as "
#                      f"vit_pretraining_checkpoint_{epoch}.pt")
#
#     # Save the whole model
#     logging.info(f"Save whole model to {save_path}...")
#     torch.save(model, Path(save_path) / f"vit_model_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.pt")
#     logging.info(f"Whole model saved to {save_path} "
#                  f"as vit_pretrained_model_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.pt")
#
#     return results


# def test(model: torch.nn.Module,
#          test_dataloader: torch.utils.data.DataLoader,
#          loss_fn: torch.nn.Module,
#          metrics: Dict[str, Any],
#          device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, List]:
#     """Tests a pytorch model.
#
#     Calculates, prints and stores evaluation metrics.
#
#     :param model: A PyTorch model to be tested.
#     :param test_dataloader: A DataLoader instance for the model to be tested on.
#     :param loss_fn: A PyTorch loss function to calculate loss on both datasets.
#     :param metrics: A Dictionary containing metrics associated with their string names.
#     :param device: The device type Pytorch will use ('cuda', 'mps' or 'cpu').
#     :return: A dictionary of testing loss as well as testing configured metrics.
#     """
#
#     test_loss, test_metrics = test_step(
#         model=model,
#         dataloader=test_dataloader,
#         loss_fn=loss_fn,
#         metrics=metrics,
#         device=device
#     )
#
#     logging.info(f"test_loss: {test_loss:.4f} | test_metrics: {test_metrics}")
#
#     return {
#         "test_loss": test_loss,
#         "test_metrics": test_metrics
#     }
