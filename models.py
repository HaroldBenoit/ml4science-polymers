import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb

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


def compute_accuracy(model, dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    X, y = next(iter(loader))
    preds = model.predict(X)
    accuracy = (preds == y).sum() * 100 / len(y)
    return accuracy


def train(train_dataset, model, test_dataset=None, num_epochs=100, batch_size=512, weight_decay=0.001, lr_rate=0.001, verbose=2):
    whole_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    config = dict(
        **whole_dataset.info(),
        num_epochs=num_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        learning_rate=lr_rate,
        optimizer='Adam',
        loss='NLLLoss'
    )
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
        
        if verbose > 1 or (verbose > 0 and epoch % 50 == 0) or epoch == num_epochs-1:
            avg_loss = np.mean(losses)
            accuracy = num_correct*100/len(train_dataset)
            wandb.log({'train_loss': avg_loss, 'train_accuracy': accuracy})
            print(f'epoch={epoch}/{num_epochs}, loss={np.mean(losses)}, accuracy={accuracy}')

    if test_dataset is not None:
        test_scores = test(test_dataset, model)
        print(f"test scores={test_scores}")
        wandb.log({f"test_{name.lower()}": score for name, score in test_scores.items()})

    wandb.finish()

    return model


def test(dataset, model, batch_size=1024):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for X, y in iter(data_loader):
            preds = model.predict(X)
            predictions = np.concatenate((predictions, preds), axis=None)
            labels = np.concatenate((labels, y), axis=None)

    names = ["Accuracy", "F1 Score", "Precision", "Recall"]
    functions = [accuracy_score, f1_score, precision_score, recall_score]

    scores = {}

    for name, func in zip(names,functions):
        score = func(labels, predictions)
        scores[name] = score
    
    return scores
