import numpy as np
from collections import namedtuple
from itertools import product
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import wandb

from pipeline import AABB245_Pipeline, PolymerDataset, train_test_split


Parameter = namedtuple('Parameter', ['name', 'value'])


class LeakyReluLSTM(nn.Module):

    def __init__(self, input_dim, output_dim=2, num_layers=1, hidden_dim=64):
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

    def __init__(self, input_dim, output_dim=2, num_layers=1, hidden_dim=64):
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


class MultiOutputLSTM(nn.Module):

    def __init__(self, input_dim, output_dim=2, num_layers=1, num_blocks=100, hidden_dim=64) -> None:
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


def train(train_dataset, model, test_dataset=None, num_epochs=100, batch_size=512, weight_decay=0.001, lr_rate=0.001, verbose=2,log=True):
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
        wandb.init(project="ml4science-polymers", config=config, entity="lucastrg")

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


def test(dataset, model, batch_size=4096):
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


def run_245(param_grid):
    parameter_space = []

    for param, values in param_grid.items():
        parameters = []

        if not isinstance(values, list) and not isinstance(values, np.ndarray):
            values = [values]

        for value in values:
            parameters.append(Parameter(name=param, value=value))

        parameter_space.append(parameters)

    for params in product(*parameter_space):
        params_dict = {param.name: param.value for param in params}
    
        pipeline = AABB245_Pipeline(num_blocks=params_dict.get('num_blocks', 8), extrema_th=params_dict.get('extrema_th', 8))
        dataset = PolymerDataset(params_dict['data_paths'], pipeline)
        model = MultiOutputLSTM(input_dim=dataset.num_features, output_dim=dataset.num_classes, num_blocks=dataset.num_blocks)
        train_data, test_data = train_test_split(dataset)
        model, metrics = train(train_dataset=train_data, test_dataset=test_data, model=model,
                               batch_size=params_dict.get('batch_size', 512), num_epochs=params_dict.get('num_epochs', 100),
                               lr_rate=params_dict.get('lr_rate', 0.001),
                               weight_decay=params_dict.get('weight_decay', 0.001),
                               verbose=0)