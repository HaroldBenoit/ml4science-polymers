import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import normal
import seaborn as sns
from scipy.signal import find_peaks_cwt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



class VanillaLSTM(nn.Module):

    def __init__(self, input_dim, output_dim=2, num_layers=1, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    
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
    
    def forward(self, X):
        outputs, _ = self.lstm(X)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1])
        outputs = self.linear1(outputs)
        outputs = F.relu(outputs)
        outputs = self.linear2(outputs)
        outputs = F.relu(outputs)
        outputs = self.linear3(outputs)
        probs = F.log_softmax(outputs, dim=1)
        return probs
    
    def predict(self, X):
        probs = self.forward(X)
        preds = torch.argmax(probs, dim=1, keepdim=False)
        return preds


def train(dataset, model, num_epochs=100, batch_size=512, weight_decay=0.001, lr_rate=0.001, verbose=1):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
        
        if verbose > 0 or (verbose > 1 and epoch % 50 == 0) or epoch == num_epochs-1:
            print(f'epoch={epoch}/{num_epochs}, loss={np.mean(losses)}, accuracy={num_correct*100/len(dataset)}')
    
    return model


def test(dataset,model,batch_size = 64):


    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for X, y in iter(data_loader):
            probs = model(X)
            preds = torch.argmax(probs, dim=1, keepdim=False)
            predictions = np.concatenate((predictions,preds), axis=None)
            labels= np.concatenate((labels,y),axis=None)


    accuracy = accuracy_score(labels,predictions)
    f1 = f1_score(labels,predictions)
    precision = precision_score(labels,predictions)
    recall = recall_score(labels,predictions)

    names =["Accuracy", "F1 Score", "Precision", "Recall"]
    functions = [accuracy_score, f1_score, precision_score, recall_score]

    for name, func in zip(names,functions):
        score = func(labels,predictions)
        print(f"{name}: {score*100:.2f}%")