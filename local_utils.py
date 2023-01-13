from sklearn.metrics import confusion_matrix, classification_report
import torch
from pathlib import Path


def report(y_pred: torch.Tensor, y: torch.Tensor) -> None:
    """Generate evaluation report

    Args:
        y_pred (torch.Tensor): Predicted classes
        y (torch.Tensor): Ground Truth
    """
    confmat = confusion_matrix(y_pred.cpu().numpy(), y.cpu().numpy())
    print(confmat)
    print()
    print(classification_report(y_pred.cpu().numpy(), y.cpu().numpy()))


def save_model(model: torch.nn.Module, save_path: str, model_name: str) -> None:
    """Save model

    Args:
        model (torch.nn.Module): Model object
        save_path (str): Folder to save
        model_name (str): Model name
    """
    save_path = Path(save_path)
    if not save_path.is_dir():
        save_path.mkdir()

    if not model_name.endswith('.pt') and not model_name.endswith('.pth'):
        model_name += '.pt'

    model_save_path = save_path / model_name
    torch.save(obj=model.state_dict(), f=model_save_path)
    print(f'Model saved at {model_save_path}')


def load_model(model: torch.nn.Module, path_to_model: str) -> torch.nn.Module:
    """Load model weights

    Args:
        model (torch.nn.Module): Model object
        path_to_model (str): Path to the model weights

    Returns:
        torch.nn.Module: Model with uploaded weights
    """

    path_to_model = Path(path_to_model)
    assert path_to_model.is_file(), f"Incorrect load path: {path_to_model}"
    assert path_to_model.__str__().endswith(".pth") or path_to_model.__str__(
    ).endswith(".pt"), "Model name should end with .pt or .pth"

    model.load_state_dict(torch.load(path_to_model))

    return model
