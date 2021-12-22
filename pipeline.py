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


def balance_data(data: np.ndarray) -> np.ndarray:
    """Make dataset balanced

    Args:
        data (np.ndarray): List of numpy arrays

    Returns:
        np.ndarray: Balanced data
    """
    balanced_data = []
    min_data_size = min([len(d) for d in data])
    for events in data:
        indices = np.random.permutation(len(events))
        balanced_data.append(events[indices[:min_data_size]])
    return balanced_data


def filter_data(data: np.ndarray, by_quantile: bool = True, min_quantile: float = 0.1, max_quantile: float = 0.9,
                min_len: int = 50, max_len: int = 10000, num_blocks: int = None) -> np.ndarray:
    """Filter out insignificant points from data.
    Removes too short or too long sequences.

    Args:
        data (np.ndarray): List of numy arrays
        by_quantile (bool, optional): Filter by quantile. Defaults to True.
        min_quantile (float, optional): Min quantile cutoff. Defaults to 0.1.
        max_quantile (float, optional): Max quantile cutoff. Defaults to 0.9.
        min_len (int, optional): Min event len cutoff. Defaults to 50.
        max_len (int, optional): Max event len cutoff. Defaults to 10000.
        num_blocks (int, optional): Number of blocks. Defaults to None.

    Returns:
        np.ndarray: Data after filtering
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


def standardize_data(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize data along axis

    Args:
        data (np.ndarray): Input data
        axis (int, optional): Axis along which to standardize. Defaults to 0.

    Returns:
        np.ndarray: Standardized data
    """
    data = np.array(data)
    data_mean = data.mean(axis=axis, keepdims=True)
    data_std = data.std(axis=axis, keepdims=True)
    data_std = np.where(np.isclose(data_std, 0), 1, data_std)
    data = (data - data_mean) / data_std
    data = np.where(np.isnan(data), 0, data)
    return data


class Pipeline:
    """
    Data processing pipeline
    """
    def __init__(self, num_blocks: int = None, block_size: int = None) -> None:
        self.data_paths = []
        self.num_blocks = num_blocks
        self.block_size = block_size

    def info(self) -> dict:
        """Return an info about the object

        Returns:
            dict: Info dict
        """
        return {
            'num_blocks': self.num_blocks,
            'block_size': self.block_size
        }

    def load(self, data_paths: List[str]) -> List[np.ndarray]:
        """Load data from given data paths

        Args:
            data_paths (List[str]): List of data paths

        Returns:
            List[np.ndarray]: List of numpy data
        """
        self.data_paths = data_paths
        return [np.load(data_path, allow_pickle=True) for data_path in data_paths]

    def process(self, raw_data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Process list of data

        Args:
            raw_data (List[np.ndarray]): List of raw data

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed data and labels
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
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Transformed data
        """
        return data
    
    def extract_features(self, event: np.ndarray) -> np.ndarray:
        """Extract features from the given event

        Args:
            event (np.ndarray): Sequence of (time, current) pairs

        Returns:
            np.ndarray: Event features
        """
        return []
    
    def balance(self, data: np.ndarray) -> np.ndarray:
        """Make data balanced

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Balanced data
        """
        return balance_data(data)
    
    def filter(self, data: np.ndarray) -> np.ndarray:
        """Filter data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Clean data
        """
        return filter_data(data)
    
    def standardize(self, data: np.ndarray) -> np.ndarray:
        """Standardize data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Standardized data
        """
        return standardize_data(data, axis=0)

    def process_event(self, event: np.ndarray) -> np.ndarray:
        """Process given event

        Args:
            event (np.ndarray): Sequence of (time, current) pairs

        Returns:
            np.ndarray: Processed event
        """
        processed_event = []
        block_size = self.block_size or int(np.ceil(len(event) / self.num_blocks))

        for i in range(self.num_blocks):
            sub_event = event[i*block_size:(i+1)*block_size]
            features = self.extract_features(sub_event)
            processed_event.append(features)

        return np.array(processed_event)


class AABB245_Pipeline(Pipeline):
    def __init__(self, num_blocks=None, block_size=None, extrema_th=0, min_event_len=50, max_event_len=10000, by_quantile=False) -> None:
        super().__init__(num_blocks=num_blocks, block_size=block_size)
        self.extrema_th = extrema_th
        self.min_event_len = min_event_len
        self.max_event_len = max_event_len
        self.by_quantile = by_quantile

    def info(self):
        parent_info = super().info()
        return {
            'extrema_th': self.extrema_th,
            'min_event_len': self.min_event_len,
            'max_event_len': self.max_event_len,
            'by_quantile': self.by_quantile,
            **parent_info
        }

    def filter(self, data):
        return filter_data(data, by_quantile=self.by_quantile, min_len=self.min_event_len, max_len=self.max_event_len)

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
        return filter_data(data, by_quantile=True,num_blocks=self.num_blocks)
class PairSingle_Pipeline(Pipeline):
    def __init__(self, num_blocks):
        super().__init__(num_blocks=num_blocks)

    def extract_features(self, event):
        features = np.array([])

        current_functions = [np.mean, np.median, np.std, np.min, np.max, len]
        row_functions = [count_extremums, max_slope, min_slope, timestamp]

        for func in current_functions:
            features = np.concatenate((features, func(event[:, 1])), axis=None)

        for func in row_functions:
            features = np.concatenate((features, func(event)), axis=None)
        
        features=np.concatenate((features, extract_fft_features(event)), axis=None)

        return features        

    def filter(self, data):
        return filter_data(data, by_quantile=True,num_blocks=self.num_blocks)

class PolymerDataset(Dataset):
    """
    Representation of Polymer dataset
    """
    def __init__(self, data_paths: List[str], pipeline: Pipeline, seed: int = 42, save_path: str = None) -> None:
        """Set up a dataset

        Args:
            data_paths (List[str]): List of data paths to load from
            pipeline (Pipeline): Processing pipeline
            seed (int, optional): Random seed. Defaults to 42.
            save_path (str, optional): Path to save data after processing. Defaults to None.
        """
        super().__init__()
        self.data_paths = data_paths
        self.pipeline = pipeline
        self.process(data_paths, pipeline, seed)
        if save_path:
            torch.save(self.data, save_path)

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