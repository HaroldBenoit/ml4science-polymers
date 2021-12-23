import pickle
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.core.shape_base import block
from torch.serialization import save
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from helpers import *

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15,7)
torch.manual_seed(1)


def balance_data(data):
    """Balance M datasets so they have the same number of datapoints across all classes

    Args:
        data (np.array of shape (M,N,X,2)): Contains M datasets each of a different class with dim (X,2) where N might not be equal amongst all M datasets

    Returns:
        data (np.array of shape (M,N',X,2)): Contains M datasets each of a different class where N' is the smallest size of all the datsets given
    """    
    balanced_data = []
    min_data_size = min([len(d) for d in data])
    for events in data:
        indices = np.random.permutation(len(events))
        balanced_data.append(events[indices[:min_data_size]])
    return balanced_data


def filter_data(data, by_quantile=True, min_quantile=0.1, max_quantile=0.9, min_len=50, max_len=10000, num_blocks=None):
    """Filters a dataset so that we only keep the rows that have a reasonable length (that are neither a misinterpreted event nor a polymer that got stuck)

    Args:
        data (np.array of shape (N,X,2)): datasets of raw events with some too long, and other too short
        by_quantile (bool, optional): Whether or not we should compute the outliers using quantiles. Defaults to True.
        min_quantile (float, optional): Quantile under which we should discard the event. Defaults to 0.1.
        max_quantile (float, optional): Quantile over which we should discard the data. Defaults to 0.9.
        min_len (int, optional): Minimum length an event should satisfy. Defaults to 50.
        max_len (int, optional): Maximum length an event should satisfy. Defaults to 10000.
        num_blocks (int, optional): Number of blocks each event has been divided by. Defaults to None.

    Returns:
        (np.array of shape (N',X,2)): datasets of filtered event with reasonable sizes
    """    
    clean_data = []
    ## such that the max function below is well defined
    if num_blocks is None:
        num_blocks = 1

    min_len = max(num_blocks + 1, min_len)

    for events in data:
        if by_quantile:
            event_lens = [len(event) for event in events]
            min_len = max(num_blocks + 1, np.quantile(event_lens, min_quantile))
            max_len = np.quantile(event_lens, max_quantile)
        clean_events = []
        for event in events:
            if len(event) > min_len and len(event) < max_len:
                clean_events.append(event)
        clean_data.append(np.array(clean_events, dtype='object'))
    return clean_data


def standardize_data(data, axis=0):
    """Standardizes the data X-> (X-mean(X))/std(X)

    Args:
        data (np.array of shape (N,B,L/B,F)): data we want to standardize where :
            -N is th number of events
            -B is the number of blocks
            -L is the length of an event
            -F is the number of features computed

        axis (int, optional): axis over which we want to compute the metrics. Defaults to 0.

    Returns:
        np.array of shape np.array of dim(N,B,L/B,F): Standardized data
    """    
    data = np.array(data)
    data_mean = data.mean(axis=axis, keepdims=True)
    data_std = data.std(axis=axis, keepdims=True)
    data_std = np.where(np.isclose(data_std, 0), 1, data_std)
    data = (data - data_mean) / data_std
    data = np.where(np.isnan(data), 0, data)
    return data


class Pipeline:
    """Pipeline used to define the transformations that will be applied to a dataset
    """    
    def __init__(self, num_blocks=None, block_size=None) -> None:
        """The constructor for a Pipeline

        Args:
            num_blocks (int, optional): . Defaults to None.
            block_size (int, optional): number of features per block. Defaults to None.
        """        
        self.data_paths = []
        self.num_blocks = num_blocks
        self.block_size = block_size

    def info(self):
        """Recap of the hyper-parameters for logging purposes

        Returns:
            dict(string -> string): dictionnary mapping for the hyper-parameters to their respective values
        """        
        return {
            'num_blocks': self.num_blocks,
            'block_size': self.block_size
        }

    def load(self, data_paths):
        """Allows to load datasets to a pipeline from a path

        Args:
            data_paths (list(str)): Path where the dataset is located

        Returns:
            np.array of shape (M,X,2): arrays of the datasets located at the provided paths
        """        
        self.data_paths = data_paths
        return [np.load(data_path, allow_pickle=True) for data_path in data_paths]

    def process(self, raw_data):
        """Chunks and processes the features of a whole dataset 

        Args:
            raw_data (np.array of shape (N,X,2)): Dataset that needs to be chunked in block and processed

        Returns:
            (np.array of shape (N,B,F), np.array of shape (N)): array of the features that will be used for classification where:
                -N is the number of events, 
                -B the number of blocks, 
                -F the number of features
                and an array of the label of each event

        """        
        data = []
        labels = []
        max_event_len = np.max([len(event) for events in raw_data for event in events])
        self.num_blocks = self.num_blocks or int(np.ceil(max_event_len / self.block_size))

        for data_index, events in enumerate(raw_data):
            for event in tqdm(events, desc=f'Processing {self.data_paths[data_index]}'):
                processed_event = self.process_event(event)
                data.append(processed_event)
                labels.append(data_index)

        return np.array(data), np.array(labels)
    
    def transform(self, data):
        """Computes transformation of the dataset (i.e. FFT, auto-correlation, etc...)

        Args:
            data (np.array of shape (N,X,2)): dataset for which we want to compute the transformations of.

        Returns:
            np.array of shape: (N,X,2+T) where T is the number of transfomations applied
        """        
        return data
    
    def extract_features(self, event):
        """Computes the features of a given event

        Args:
            event (np.array of shape (B,N//B,2+T)): event for which we want to compute the features for where :
                -B is the number of blocks
                -N is the size of the event
                -T is the number of transformations applied to the data

        Returns:
            np.array of shape (B,F): processed features for the event where :
                -B is the number of blocks
                -F is the number of features 
        """        
        return []
    
    def balance(self, data):
        """Balances dataset so each class has the same number of events to classify

        Args:
            data (np.array of shape (M,N,X,2)): dataset we want to balance where :
                -M is the number of source datasets/classes
                -N is the number of events (varies for each class)
                -X is the length of each event (whch varies for each event)


        Returns:
            data (np.array of shape (M,N',X,2)): Contains M datasets each of a different class where N' is the smallest size of all the datsets given
        """        
        return balance_data(data)
    
    def filter(self, data):
        """filters the events of a given datasets

        Args:
            data (np.array of shape (N,X,2)): datasets of raw events with some too long, and other too short

        Returns:
            (np.array of shape (N',X,2)): datasets of filtered event with reasonable sizes
        """        
        return filter_data(data)
    
    def standardize(self, data):
        """standardizes the data of a given dataset

        Args:
            data (np.array of shape (N,X,2+F)): array of the features we want to standardize

        Returns:
            (np.array of shape (N,X,2+F): array of the standardized data
        """        
        return standardize_data(data, axis=0)

    def process_event(self, event):
        """divides an event in blocks, abd processed the feature for each block

        Args:
            event (np.array(N,2)): event of N measurements of both (current, time)

        Returns:
            np.array of shape (B,F): features for each block of the given event
        """        
        processed_event = []
        block_size = self.block_size or int(np.ceil(len(event) / self.num_blocks))

        for i in range(self.num_blocks):
            sub_event = event[i*block_size:(i+1)*block_size]
            features = self.extract_features(sub_event)
            processed_event.append(features)

        return np.array(processed_event)

class AABB245_Pipeline(Pipeline):
    """Pipeline for the multi-class classification

    Args:
        Pipeline ([type]): [description]
    """    
    def __init__(self, num_blocks=None, block_size=None, extrema_th=0, min_event_len=50, max_event_len=10000, by_quantile=False) -> None:
        """

        Args:
            num_blocks ([type], optional): [description]. Defaults to None.
            block_size ([type], optional): [description]. Defaults to None.
            extrema_th (int, optional): [description]. Defaults to 0.
            min_event_len (int, optional): [description]. Defaults to 50.
            max_event_len (int, optional): [description]. Defaults to 10000.
            by_quantile (bool, optional): [description]. Defaults to False.
        """        
        super().__init__(num_blocks=num_blocks, block_size=block_size)
        self.extrema_th = extrema_th
        self.min_event_len = min_event_len
        self.max_event_len = max_event_len
        self.by_quantile = by_quantile

    def info(self):
        """Recap of the hyper-parameters for logging purposes

        Returns:
            dict(string -> string): dictionnary mapping for the hyper-parameters to their respective values
        """         
        parent_info = super().info()
        return {
            'extrema_th': self.extrema_th,
            'min_event_len': self.min_event_len,
            'max_event_len': self.max_event_len,
            'by_quantile': self.by_quantile,
            **parent_info
        }

    def filter(self, data):
        """filters the events of a given datasets

        Args:
            data (np.array of shape (N,X,2)): datasets of raw events with some too long, and other too short

        Returns:
            (np.array of shape (N',X,2)): datasets of filtered event with reasonable sizes
        """         
        return filter_data(data, by_quantile=self.by_quantile, min_len=self.min_event_len, max_len=self.max_event_len)

    def extract_features(self, event):
        """extracts the features for the multi-class classification. With basic features (mean, std, etc...) and features computed on the Fast Fourrier Transform (fundamental frequencies, etc...)

        Args:
            event (np.array of shape(N,X,2)): datasets of N events of X measurements (different for each) with a timestamp and a relative current for each event.

        Returns:
            np.array of shape (N,B,F): features of each divided event
        """        
        basic_features = extract_basic_features(event)
        extrema_features = extract_extrema_features(event, extrema_th=self.extrema_th)
        fft_features = extract_fft_features(event)
        return np.concatenate([basic_features, extrema_features, fft_features])


class AA0066_Pipeline(Pipeline):
    """Pipeline for the backbone classification

    Args:
        Pipeline ([type]): [description]
    """    
    def __init__(self, num_blocks):
        super().__init__(num_blocks=num_blocks)

    def extract_features(self, event):
        """extracts the features for the backbone classification. With basic features (mean, std, etc...)

        Args:
            event (np.array of shape(N,X,2)): datasets of N events of X measurements (different for each) with a timestamp and a relative current for each event.

        Returns:
            np.array of shape (N,B,F): features of each divided event
        """          
        features = np.array([])

        current_functions = [np.mean, np.median, np.std, np.min, np.max, len]
        row_functions = [count_extremums, max_slope, min_slope]

        for func in current_functions:
            features = np.concatenate((features, func(event[:, 1])), axis=None)

        for func in row_functions:
            features = np.concatenate((features, func(event)), axis=None)

        return features        

    def filter(self, data):
        """filters the events of a given datasets

        Args:
            data (np.array of shape (N,X,2)): datasets of raw events with some too long, and other too short

        Returns:
            (np.array of shape (N',X,2)): datasets of filtered event with reasonable sizes
        """    
        return filter_data(data, by_quantile=True,num_blocks=self.num_blocks)
class PairSingle_Pipeline(Pipeline):
    """Pipeline for the sequence classification

    Args:
        Pipeline ([type]): [description]
    """    
    def __init__(self, num_blocks):
        super().__init__(num_blocks=num_blocks)

    def extract_features(self, event):
        """extracts the features for the sequence classification. With basic features (mean, std, etc...) and features computed on the Fast Fourrier Transform (fundamental frequencies, etc...)

        Args:
            event (np.array of shape(N,X,2)): datasets of N events of X measurements (different for each) with a timestamp and a relative current for each event.

        Returns:
            np.array of shape (N,B,F): features of each divided event
        """   
        features = np.array([])

        current_functions = [np.mean, np.median, np.std, np.min, np.max, len]
        row_functions = [count_extremums, max_slope, min_slope]        
        for func in current_functions:
            features = np.concatenate((features, func(event[:, 1])), axis=None)

        for func in row_functions:
            features = np.concatenate((features, func(event)), axis=None)
        
        features=np.concatenate((features, extract_fft_features(event)), axis=None)

        return features        

    def filter(self, data):
        """filters the events of a given datasets

        Args:
            data (np.array of shape (N,X,2)): datasets of raw events with some too long, and other too short

        Returns:
            (np.array of shape (N',X,2)): datasets of filtered event with reasonable sizes
        """    
        return filter_data(data, by_quantile=True,num_blocks=self.num_blocks)

class PolymerDataset(Dataset):
    """
    Representation of Polymer dataset
    """
    def __init__(self, data_paths: List[str], pipeline: Pipeline, seed: int = 42, save_path: str = None, load_path: str = None) -> None:
        """Set up a dataset

        Args:
            data_paths (List[str]): List of data paths to load from
            pipeline (Pipeline): Processing pipeline
            seed (int, optional): Random seed. Defaults to 42.
            save_path (str, optional): Path to save data after processing. Defaults to None.
            load_path (str, optional): Path to load data from. Defaults to None.
        """
        super().__init__()
        self.data_paths = data_paths
        self.pipeline = pipeline
        if load_path:
            self.data = torch.load(f'{load_path}_data.pt')
            self.labels = torch.load(f'{load_path}_labels.pt')
        else:
            self.process(data_paths, pipeline, seed)
        if save_path:
            torch.save(self.data, f'{save_path}_data.pt')
            torch.save(self.labels, f'{save_path}_labels.pt')

    def info(self) -> dict:
        """Return an info about the dataset

        Returns:
            dict: Info dict
        """
        return {
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'data': self.data_paths,
            **self.pipeline.info()
        }

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

    def process(self, data_paths: List[str], pipeline: Pipeline, seed: int = 42):
        """Load and prepare data as a torch dataset

        Args:
            data_paths (List[str]): List of data paths
            pipeline (Pipeline): Processing pipeline
            seed (int, optional): Random seed. Defaults to 42.
        """
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

        self.data = data
        self.labels = labels

        
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

        return self
