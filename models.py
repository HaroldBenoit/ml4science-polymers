from typing import Callable
import inspect
import numpy as np
from collections import namedtuple, defaultdict
from itertools import product
from functools import partial
import torch
from torch.utils.data import DataLoader, Subset, Dataset, dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import wandb

from pipeline import AABB245_Pipeline, PolymerDataset, train_test_split


Parameter = namedtuple('Parameter', ['name', 'value'])


class LeakyReluLSTM(nn.Module):

    def __init__(self, input_dim, output_dim=2, num_layers=1, hidden_dim=64, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.af1 = nn.LeakyReLU()
        self.af2 = nn.LeakyReLU()
        
    def info(self):
        return {
            'model_name': ' LeakyReluLSTM',
            'lstm_input_dim': self.lstm.input_size,
            'lstm_hidden_dim': self.lstm.hidden_size,
            'lstm_num_layers': self.lstm.num_layers
        }
    
    def forward(self, X):
        outputs, _ = self.lstm(X)
        outputs = outputs[:, -1, :]

        outputs = self.linear1(self.af1(outputs))
        outputs = self.linear2(self.af2(outputs))

        probs = F.log_softmax(outputs, dim=1)

        return probs

    def predict(self, X):
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds


class VanillaLSTM(nn.Module):

    def __init__(self, input_dim, output_dim=2, num_layers=1, hidden_dim=64, *args, **kwargs):
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

    def __init__(self, input_dim, output_dim=2, num_layers=1, num_blocks=100, hidden_dim=64, *args, **kwargs) -> None:
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

    def forward(self, X):
        outputs, _ = self.lstm(X)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1])
        outputs = self.linear1(F.relu(outputs))
        outputs = self.linear2(F.relu(outputs))
        outputs = self.linear3(F.relu(outputs))
        probs = F.log_softmax(outputs, dim=1)
        return probs
    
    def predict(self, X):
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds


def train(train_dataset, model, test_dataset=None, num_epochs=100, batch_size=512, weight_decay=0.001, lr_rate=0.001, verbose=2, log=True, *args, **kwargs):
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


def test(dataset, model, batch_size=4096, *args, **kwargs):
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


def kfold_cv_iter(dataset, k=5, seed=1):
    num_samples = len(dataset)
    fold_size = int(num_samples / k)
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = list(set(range(num_samples)) - set(test_indices))
        yield Subset(dataset, train_indices), Subset(dataset, test_indices)


def cross_validate(dataset, model, train_fn, test_fn, k_fold=5, seed=42):
    cv_metrics = defaultdict(int)

    for train_dataset, test_dataset in kfold_cv_iter(dataset, k=k_fold, seed=seed):
        _ = train_fn(train_dataset, model=model)
        metrics = test_fn(test_dataset, model=model)
        for metric_key, metric_value in metrics.items():
            cv_metrics[metric_key] += metric_value

    return {metric_key: metric_value / k_fold for metric_key, metric_value in cv_metrics.items()}


def grid_search_cv(data_paths, param_grid, model_fn, train_fn, test_fn, transform_fn,
                   scoring: str = 'accuracy', k_fold=5, seed=1):
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
        
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    return {param.name: param.value for param in best_params}, best_metrics