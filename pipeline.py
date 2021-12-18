from abc import ABC, abstractmethod

import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils import data

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from helpers import *

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15,7)
torch.manual_seed(1)


def balance_data(data):
    balanced_data = []
    min_data_size = min([len(d) for d in data])
    for events in data:
        indices = np.random.permutation(len(events))
        balanced_data.append(events[indices[:min_data_size]])
    return balanced_data


def filter_data(data, by_quantile=True, min_quantile=0.1, max_quantile=0.9, min_len=50, max_len=10000):
    clean_data = []
    for events in data:
        if by_quantile:
            event_lens = [len(event) for event in events]
            min_len = np.quantile(event_lens, min_quantile)
            max_len = np.quantile(event_lens, max_quantile)
        clean_events = []
        for event in events:
            if len(event) > min_len and len(event) < max_len:
                clean_events.append(event)
        clean_data.append(np.array(clean_events, dtype='object'))
    return clean_data


def standardize_data(data, axis=0):
    data = np.array(data)
    data_mean = data.mean(axis=axis, keepdims=True)
    data_std = data.std(axis=axis, keepdims=True)
    data_std = np.where(np.isclose(data_std, 0), 1, data_std)
    data = (data - data_mean) / data_std
    data = np.where(np.isnan(data), 0, data)
    return data


class Pipeline:
    def __init__(self, num_blocks=None, block_size=None) -> None:
        self.data_paths = []
        self.num_blocks = num_blocks
        self.block_size = block_size

    def load(self, data_paths):
        self.data_paths = data_paths
        return [np.load(data_path, allow_pickle=True) for data_path in data_paths]

    def process(self, raw_data):
        data = []
        labels = []
        max_event_len = np.max([len(event) for events in raw_data for event in events])
        self.num_blocks = self.num_blocks or int(np.ceil(max_event_len / self.block_size))

        for data_index, events in enumerate(raw_data):
            for event in tqdm(events, desc=f'Processing {self.data_paths[data_index]}'):
                processed_event = self.process_event(event)
                data.append(processed_event)
                labels.append(data_index)

        return data, labels
    
    def transform(self, data):
        return data
    
    def extract_features(self, event):
        return []
    
    def balance(self, data):
        return balance_data(data)
    
    def filter(self, data):
        return filter_data(data)
    
    def standardize(self, data):
        return standardize_data(data, axis=0)

    def process_event(self, event):
        processed_event = []
        block_size = self.block_size or int(np.ceil(len(event) / self.num_blocks))
        for i in range(self.num_blocks):
            sub_event = event[i*block_size:(i+1)*block_size]
            features = self.extract_features(sub_event)
            processed_event.append(features)
        return np.array(processed_event)


class AABB245_Pipeline(Pipeline):
    def __init__(self, num_blocks=None, block_size=None, extrema_th=0, min_event_len=50, max_event_len=10000) -> None:
        super().__init__(num_blocks=num_blocks, block_size=block_size)
        self.extrema_th = extrema_th
        self.min_event_len = min_event_len
        self.max_event_len = max_event_len

    def filter(self, data):
        return filter_data(data, by_quantile=False, min_len=self.min_event_len, max_len=self.max_event_len)

    def extract_features(self, event):
        basic_features = extract_basic_features(event)
        extrema_features = extract_extrema_features(event, extrema_th=self.extrema_th)
        fft_features = extract_fft_features(event)
        return np.concatenate([basic_features, extrema_features, fft_features])


class AA0066_Pipeline(Pipeline):
    def __init__(self, num_blocks):
        super().__init__(num_blocks=num_blocks)

    def extract_features(self, event):
        features = np.array([])

        current_functions = [np.mean, np.median, np.std, np.min, np.max, len]
        row_functions = [count_extremums, max_slope, min_slope]

        for func in current_functions:
            features = np.concatenate((features, func(event[:, 1])), axis=None)

        for func in row_functions:
            features = np.concatenate((features, func(event)), axis=None)

        return features        

    def filter(self, data):
        return filter_data(data, by_quantile=True)


class PolymerDataset(Dataset):
    def __init__(self, data_paths, pipeline, seed=42, save_path=None):
        super().__init__()
        self.data_paths = data_paths
        self.process(data_paths, pipeline, seed)
        if save_path:
            torch.save(self.data, save_path)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @property
    def num_features(self):
        return self.data.shape[2]
    
    @property
    def num_classes(self):
        return len(torch.unique(self.labels))

    @property
    def num_blocks(self):
        return self.data.shape[1]

    def process(self, data_paths, pipeline, seed=42):    
        np.random.seed(seed)  

        # Load data 
        data = pipeline.load(data_paths)

        # Make data balanced
        data = pipeline.balance(data)

        # Remove too short and too long events
        data = pipeline.filter(data)
        data, labels = pipeline.process(data)

        # Standardize
        data = pipeline.standardize(data)

        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

        return self