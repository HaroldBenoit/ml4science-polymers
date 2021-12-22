import numpy as np
from typing import Generator, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import random_split, Dataset, Subset



def split_in_k(y: np.ndarray, row: np.ndarray, k: int, seed: int = 1) -> Generator:
    """Split given data in k parts and return an iterator over the parts

    Args:
        y (np.ndarray): Data
        row (np.ndarray): Data row
        k (int): Number of parts
        seed (int, optional): Random seed. Defaults to 1.

    Yields:
        Generator: Iterator over parts
    """
    chunk_size = len(row)//k
    np.random.seed(seed)
    indices = np.random.permutation(len(row))
    
    for i in range(k):
        index=indices[i * chunk_size: (i + 1) * chunk_size]
        yield y, row[index]


def count_extremums(row: np.ndarray) -> int:
    """Count number of extrema in the row of data

    Args:
        row (np.ndarray): Data row

    Returns:
        int: Number of extrema
    """
    current = row[:, 1]
    if len(current) < 3:
        return 0
    increasing = current[0] < current[1]
    tmp = current[0]
    counter = 0
    values = []
    for i in current[1:]:
        if increasing and i < tmp:
            increasing = False
            counter += 1
            values.append(i)
        elif not increasing and i > tmp:
            increasing = True
            counter += 1
            values.append(i)

        tmp = i
    return counter

def timestamp(row: np.ndarray) -> Tuple:
    """Return timestamp of the row

    Args:
        row (np.ndarray): Data row

    Returns:
        Tuple: Tuple of timestamp values
    """
    return row[1:0], row[-1:0], row[1:0]-row[-1:0]

def max_slope(row: np.ndarray) -> float:
    """Calculate maximum slope

    Args:
        row (np.ndarray): Data row

    Returns:
        float: Maximum slope
    """
    maxslope = -1000
    for i in range(len(row)-1):
        if (row[i+1, 0] - row[i, 0]) != 0 and (row[i+1, 1] - row[i, 1]) / (row[i+1, 0] - row[i, 0]) > maxslope:
            maxslope = (row[i+1, 1] - row[i, 1]) / (row[i+1, 0] - row[i, 0])
    return maxslope


def min_slope(row: np.ndarray) -> float:
    """Calculate minimum slope

    Args:
        row (np.ndarray): Data row

    Returns:
        float: Minimum slope
    """
    minslope = 1000
    for i in range(len(row)-1):
        if (row[i+1, 0] - row[i, 0]) != 0 and (row[i+1, 1] - row[i, 1]) / (row[i+1, 0] - row[i, 0]) < minslope:
            minslope = (row[i+1, 1] - row[i, 1]) / (row[i+1, 0] - row[i, 0])
    return minslope


def find_extrema(event: np.ndarray, extrema_th: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Find extrema of a sequence

    Args:
        event (np.ndarray): Sequence of (time, current) pairs
        extrema_th (int, optional): Min threshold on the 2 nearby extrema difference. Defaults to 2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of peaks and lows
    """
    extrema = []
    peaks = []
    lows = []
    extrema.append(event[0])

    for i in range(1, len(event)-1):
        this_current = event[i, 1]
        prev_current = event[i-1, 1]
        next_current = event[i+1, 1]
        prev_change = this_current - prev_current
        next_change = this_current - next_current
        if prev_change * next_change >= 0:
            extrema_change = np.abs(this_current - extrema[-1][1])
            if extrema_change > extrema_th:
                extrema.append(event[i])
                if prev_change > 0:
                    peaks.append(event[i])
                else:
                    lows.append(event[i])

    return np.array(peaks), np.array(lows)


def plot_data(data: np.ndarray, plot_extrema: bool = False, plot_fft: bool = False, extrema_th: int = 0) -> None:
    """Plot signal data

    Args:
        data (np.ndarray): Signal data
        plot_extrema (bool, optional): Plot extrema points. Defaults to False.
        plot_fft (bool, optional): Plot FFT amplitudes. Defaults to False.
        extrema_th (int, optional): Extrema threshold. Defaults to 0.
    """
    if data.ndim > 1:
        data = np.expand_dims(data, axis=0)

    k = len(data)
    fig, axes = plt.subplots(k, 1, figsize=(30, 30), sharex=True, sharey=True)

    for i in range(k):
        event = data[i]
        time = event[:, 0]
        current = event[:, 1]
        ax = axes[i] if isinstance(axes, np.ndarray) else axes
        g = sns.lineplot(x=time, y=current, ax=ax)
        g.set_xlabel('Time [ms]')
        g.set_ylabel('Current')

        if plot_extrema:
            peaks, lows = find_extrema(event, extrema_th=extrema_th)
            peak_times = peaks[:, 0]
            peak_currents = peaks[:, 1]
            low_times = lows[:, 0]
            low_currents = lows[:, 1]
            sns.scatterplot(x=peak_times, y=peak_currents, ax=ax, s=100, color='blue')
            sns.scatterplot(x=low_times, y=low_currents, ax=ax, s=100, color='red')
        
        if plot_fft:
            fft = np.fft.fft(event[:, 1])
            amplitudes = np.abs(fft)
            fft_features = extract_fft_features(event)
            ax.set_yscale('log')
            ax.plot(time, amplitudes, color='grey')
            ax.axvspan(fft_features[-2], fft_features[-1], color='yellow')


def extract_fft_features(event: np.ndarray, diff_th: int = 10) -> np.ndarray:
    """Extract FFT features from an event

    Args:
        event (np.ndarray): Sequence of (time, current) pairs
        diff_th (int, optional): Difference threshold to determine FFT dwell. Defaults to 10.

    Returns:
        np.ndarray: FFT features
    """
    features = {
        'max_amp': 0,
        'min_amp': 0,
        'mean_amp': 0,
        'std_amp': 0,
        'median_amp': 0,
        'dwell_start': 0,
        'dwell_end': 0
    }

    if len(event) > 0:
        fft = np.fft.fft(event[:, 1])
        time = event[:, 0]
        amplitudes = np.abs(fft)
        dwells = []
        dwell = []

        for i in range(1, len(amplitudes)):
            diff = amplitudes[i]-amplitudes[i-1]
            if diff < diff_th:
                dwell.append(time[i])
            else:
                dwells.append(dwell)
                dwell = []
        
        features['max_amp'] = np.max(amplitudes)
        features['min_amp'] = np.min(amplitudes)
        features['mean_amp'] = np.mean(amplitudes)
        features['std_amp'] = np.std(amplitudes)
        features['median_amp'] = np.median(amplitudes)

        if len(dwells) > 0:
            longest_dwell = max(dwells, key=lambda d: len(d))
            if len(longest_dwell) > 0:
                features['dwell_start'] = longest_dwell[0]
                features['dwell_end'] = longest_dwell[-1]
    
    return np.array(list(features.values()))


def extract_extrema_features(event: np.ndarray, extrema_th: int = 0) -> np.ndarray:
    """Extract extrema features from an event

    Args:
        event (np.ndarray): Sequence of (time, current) pairs
        extrema_th (int, optional): Extrema threshold. Defaults to 0.

    Returns:
        np.ndarray: Extrema features
    """
    features = {
        'num_peaks': 0,
        'num_lows': 0,
        'mean_peaks': 0,
        'std_peaks': 0,
        'mean_lows': 0,
        'std_lows': 0
    }

    if len(event) > 0:
        peaks, lows = find_extrema(event, extrema_th=extrema_th)
        features['num_peaks'] = len(peaks)
        features['num_lows'] = len(lows)
        features['mean_peaks'] = np.mean(peaks) if len(peaks) > 0 else 0
        features['std_peaks'] = np.std(peaks) if len(peaks) > 0 else 0
        features['mean_lows'] = np.mean(lows) if len(lows) > 0 else 0
        features['std_lows'] = np.std(lows) if len(lows) > 0 else 0
    
    return np.array(list(features.values()))


def extract_basic_features(event: np.ndarray) -> np.ndarray:
    """Extract basic features from an event

    Args:
        event (np.ndarray): Sequence of (time, current) pairs 

    Returns:
        np.ndarray: Basic features
    """
    features = {
        'event_len': 0, 
        'dwell_time': 0,
        'max_current': 0, 
        'min_current': 0, 
        'mean_current': 0,
        'std_current': 0,
        'median_current': 0
    }
    if len(event) > 0:
        features['event_len'] = len(event)
        features['dwell_time'] = event[-1][0]-event[0][0]
        features['max_current'] = np.max(event[:, 1])
        features['min_current'] = np.min(event[:, 1])
        features['mean_current'] = np.mean(event[:, 1])
        features['std_current'] = np.std(event[:, 1])
        features['median_current'] = np.median(event[:, 1])

    return np.array(list(features.values()))


def train_test_split(dataset: Dataset, test_size: float = 0.2) -> Tuple[Subset, Subset]:
    """Split the dataset into train and test data

    Args:
        dataset (Dataset): Full torch dataset
        test_size (float, optional): Test data size. Defaults to 0.2.

    Returns:
        Tuple[Subset, Subset]: Train and test data
    """
    test_size = int(test_size * len(dataset))
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    return train_data, test_data