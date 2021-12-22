from typing import Callable, Generator, List, Dict, Any, Tuple
import inspect
import numpy as np
from collections import namedtuple, defaultdict
from itertools import product
from functools import partial
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import wandb


Parameter = namedtuple('Parameter', ['name', 'value'])


class PReLULSTM(nn.Module):

    def __init__(self, input_dim: int, output_dim: int = 2, num_layers: int = 1, hidden_dim: int = 64, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.af1 = nn.PReLU()
        self.af2 = nn.PReLU()
        
    def info(self):
        return {
            'model_name': ' LeakyReluLSTM',
            'lstm_input_dim': self.lstm.input_size,
            'lstm_hidden_dim': self.lstm.hidden_size,
            'lstm_num_layers': self.lstm.num_layers
        }
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(X)
        outputs = outputs[:, -1, :]

        outputs = self.linear1(self.af1(outputs))
        outputs = self.linear2(self.af2(outputs))

        probs = F.log_softmax(outputs, dim=1)

        return probs

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds


class VanillaLSTM(nn.Module):

    def __init__(self, input_dim: int, output_dim: int = 2, num_layers: int = 1, hidden_dim: int = 64, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def info(self):
        return {
            'model_name': 'VanillaLSTM',
            'lstm_input_dim': self.lstm.input_size,
            'lstm_hidden_dim': self.lstm.hidden_size,
            'lstm_num_layers': self.lstm.num_layers
        }
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(X)
        outputs = outputs[:, -1, :]

        outputs = self.linear1(F.relu(outputs))
        outputs = self.linear2(F.relu(outputs))

        probs = F.log_softmax(outputs, dim=1)

        return probs

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds


class AugmentedLSTM(nn.Module):

    def __init__(self, input_dim, output_dim=2, num_layers=2, hidden_dim=64, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def info(self):
        return {
            'model_name': 'AugmentedLSTM',
            'lstm_input_dim': self.lstm.input_size,
            'lstm_hidden_dim': self.lstm.hidden_size,
            'lstm_num_layers': self.lstm.num_layers
        }
    
    def forward(self, X):
        outputs, _ = self.lstm(X)
        outputs = outputs[:, -1, :]

        outputs = self.linear1(F.relu(outputs))
        outputs = self.linear2(F.relu(outputs))

        probs = F.log_softmax(outputs, dim=1)

        return probs

    def predict(self, X):
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds


class MultiOutputLSTM(nn.Module):

    def __init__(self, input_dim: int, output_dim: int = 2, num_layers: int = 1, num_blocks: int = 100, hidden_dim: int = 64, *args, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=1, batch_first=True)
        self.linear1 = nn.Linear(num_blocks, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def info(self):
        return {
            'model_name': 'MultiOutputLSTM',
            'lstm_input_dim': self.lstm.input_size,
            'lstm_hidden_dim': self.lstm.hidden_size,
            'linear_hidden_dim': self.linear1.out_features,
            'lstm_num_layers': self.lstm.num_layers
        }

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(X)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1])
        outputs = self.linear1(F.relu(outputs))
        outputs = self.linear2(F.relu(outputs))
        outputs = self.linear3(F.relu(outputs))
        probs = F.log_softmax(outputs, dim=1)
        return probs
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds


def train(train_dataset: Dataset, model: nn.Module, test_dataset: Dataset = None, num_epochs: int = 100, batch_size: int = 512,
          weight_decay: float = 0.001, lr_rate: float = 0.001, verbose: int = 2, log: bool = True, *args, **kwargs):
    """Train a neural network model

    Args:
        train_dataset (Dataset): Train dataset
        model (nn.Module): Model to train
        test_dataset (Dataset, optional): Test dataset. Defaults to None.
        num_epochs (int, optional): Number of epochs. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 512.
        weight_decay (float, optional): Weight decay for optimizer. Defaults to 0.001.
        lr_rate (float, optional): Learning rate for the model. Defaults to 0.001.
        verbose (int, optional): Verbosity level. 2 is highest and 0 is lowest. Defaults to 2.
        log (bool, optional): Whether to log results to wandb. Defaults to True.

    Returns:
        [type]: [description]
    """
    full_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    config = dict(
        **full_dataset.info(),
        **model.info(),
        num_epochs=num_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        learning_rate=lr_rate,
        optimizer='Adam',
        loss='NLLLoss',
        train_size=len(train_dataset),
        test_size=len(test_dataset) if test_dataset is not None else None
    )

    if log:
        wandb.init(project="ml4science-polymers", config=config)

    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        num_correct = 0
        losses = []

        for X, y in iter(data_loader):
            model.zero_grad()
            probs = model(X)
            loss = loss_function(probs, y)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(probs, dim=1, keepdim=False)
            num_correct += (preds == y).sum()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        accuracy = num_correct*100/len(train_dataset)
        if log:
            wandb.log({'train_loss': avg_loss, 'train_accuracy': accuracy})

        if verbose > 1 or (verbose > 0 and (epoch % 50 == 0 or epoch == num_epochs-1)):
            print(f'epoch={epoch}/{num_epochs}, loss={np.mean(losses):.6f}, accuracy={accuracy:.4f}')

        test_metrics = {}
    
        if test_dataset is not None:
            test_metrics = test(test_dataset, model)

            if verbose > 1 or (verbose > 0 and (epoch % 50 == 0 or epoch == num_epochs-1)):
                print(f"epoch={epoch}/{num_epochs}, test_accuracy={test_metrics['accuracy']*100:.4f}")

            if log:
                wandb.log({f"test_{name.lower()}": metric for name, metric in test_metrics.items()})

    if log:
        wandb.finish(quiet=True)

    return model, test_metrics


def test(dataset: Dataset, model: nn.Module, batch_size: int = 4096, *args, **kwargs) -> dict:
    """Test model on a dataset and report metrics

    Args:
        dataset (Dataset): Test dataset
        model (nn.Module): Model to test
        batch_size (int, optional): Batch size. Defaults to 4096.

    Returns:
        dict: Metrics as a dictionary
    """
    full_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    multiclass = full_dataset.num_classes > 2
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for X, y in iter(data_loader):
            preds = model.predict(X)
            predictions = np.concatenate((predictions, preds), axis=None)
            labels = np.concatenate((labels, y), axis=None)

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1_score': f1_score(labels, predictions, average='weighted' if multiclass else 'binary'),
        'precision': precision_score(labels, predictions, average='weighted' if multiclass else 'binary'),
        'recall': recall_score(labels, predictions, average='weighted' if multiclass else 'binary'),
        'confusion_matrix': confusion_matrix(labels, predictions, normalize='true').round(3)
    }
    
    return metrics


def kfold_cv_iter(dataset: Dataset, k: int = 5, seed: int = 1) -> Generator:
    """Iterate over K folds of data

    Args:
        dataset (Dataset): Input dataset
        k (int, optional): Number of folds. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 1.

    Yields:
        Generator: [description]
    """
    num_samples = len(dataset)
    fold_size = int(num_samples / k)
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = list(set(range(num_samples)) - set(test_indices))
        yield Subset(dataset, train_indices), Subset(dataset, test_indices)


def cross_validate(dataset: Dataset, model: nn.Module, train_fn: Callable, test_fn: Callable, k_fold: int = 5, seed: int = 42) -> dict:
    """Cross validate the model on the given dataset.

    Args:
        dataset (Dataset): Input dataset
        model (nn.Module): Model to evaluate
        train_fn (Callable): Train function
        test_fn (Callable): Test function
        k_fold (int, optional): Number of CV folds. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        dict: [description]
    """
    cv_metrics = defaultdict(int)

    for train_dataset, test_dataset in kfold_cv_iter(dataset, k=k_fold, seed=seed):
        _ = train_fn(train_dataset, model=model)
        metrics = test_fn(test_dataset, model=model)
        for metric_key, metric_value in metrics.items():
            cv_metrics[metric_key] += metric_value

    return {metric_key: metric_value / k_fold for metric_key, metric_value in cv_metrics.items()}


def grid_search_cv(data_paths: List[str], param_grid: Dict[str, Any], model_fn: Callable, train_fn: Callable,
                   test_fn: Callable, transform_fn: Callable, scoring: str = 'accuracy', k_fold: int = 5, seed: int = 1) -> Tuple[dict, dict]:
    """Grid search over the given parameter space and select params and metrics for best performing models.

    Args:
        data_paths (List[str]): List of data paths
        param_grid (Dict[str, Any]): Parameter grid to search in
        model_fn (Callable): Model builder function
        train_fn (Callable): Training function
        test_fn (Callable): Testing function
        transform_fn (Callable): Data transformation function.
        scoring (str, optional): Scoring metric to use. Defaults to 'accuracy'.
        k_fold (int, optional): Number of CV folds. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        Tuple[dict, dict]: [description]
    """
    best_score = None
    best_params = None
    best_metrics = None
    parameter_space = []

    for param, values in param_grid.items():
        parameters = []

        if not isinstance(values, list) and not isinstance(values, np.ndarray):
            values = [values]

        for value in values:
            # Convert other sequences to tuple to make the parameter accesible to be used as a dictionary key
            parameters.append(Parameter(name=param, value=value if np.isscalar(value) else tuple(value)))

        parameter_space.append(parameters)
    
    transformations = {}
    transform_params = list(inspect.signature(transform_fn).parameters.keys()) if transform_fn else None

    for params in product(*parameter_space):
        params_dict = {param.name: param.value for param in params}

        # Check if the transformation already exists with these parameters and avoid extra computation
        common_params = tuple([param for param in params if param.name in transform_params])
        dataset = transformations.get(common_params)
        if dataset is None:
            # Store transformations for later use
            dataset = transform_fn(data_paths, **params_dict)
            transformations[common_params] = dataset

        model = model_fn(dataset=dataset, **params_dict)
        train_fn_partial = partial(train_fn, **params_dict)
        test_fn_partial = partial(test_fn, **params_dict)

        metrics = cross_validate(dataset=dataset, model=model, train_fn=train_fn_partial, test_fn=test_fn_partial,
                                 k_fold=k_fold, seed=seed)

        score = metrics[scoring]
        
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    return {param.name: param.value for param in best_params}, best_metrics