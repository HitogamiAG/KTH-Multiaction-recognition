import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple


def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Train step for one iteration

    Args:
        model (nn.Module): NN Model
        dataloader (DataLoader): Custom dataloader
        loss_fn (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer algorithm
        device (torch.device): Device (CPU/GPU)

    Returns:
        Tuple[float, float]: Loss value and accuract value for the train step
    """

    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)

    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Validation step for one iteration

    Args:
        model (nn.Module): NN Model
        dataloader (DataLoader): Custom dataloader
        loss_fn (nn.Module): Loss function
        device (torch.device): Device (CPU/GPU)

    Returns:
        Tuple[float, float]: Loss value and accuracy value for the validation step
    """

    val_loss, val_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            val_loss += loss.item()

            y_class_pred = torch.argmax(torch.softmax(y_pred, axis=1), axis=1)
            val_acc += (y_class_pred == y).sum().item()

    val_loss /= len(dataloader)
    val_acc /= len(dataloader.dataset)

    return val_loss, val_acc


def evaluate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Test performance of the model

    Args:
        model (nn.Module): NN Model
        dataloader (DataLoader): Custom dataloader
        loss_fn (nn.Module): Loss function
        device (torch.device): Device (CPU/GPU)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, float, float]: Tensors of predicted and true classes, loss value and accuracy value for the test
    """

    test_loss, test_acc = 0, 0
    model.eval()

    y_pred_concatenated = torch.IntTensor().to(device)
    y_concatenated = torch.IntTensor().to(device)

    with torch.inference_mode():
        for X, y in dataloader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_loss += loss.item()

            y_class_pred = torch.argmax(torch.softmax(y_pred, axis=1), axis=1)
            y_pred_concatenated = torch.cat(
                [y_pred_concatenated, y_class_pred])
            y_concatenated = torch.cat([y_concatenated, y])
            test_acc += (y_class_pred == y).sum().item()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader.dataset)

    return y_pred_concatenated, y_concatenated, test_loss, test_acc