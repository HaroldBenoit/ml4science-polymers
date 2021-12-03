import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks_cwt

def find_extrema(event, diff_threshold=2):
    extrema = []
    extrema.append(event[0])

    for i in range(1, len(event)-1):
        this_current = event[i, 1]
        prev_current = event[i-1, 1]
        next_current = event[i+1, 1]
        prev_change = this_current - prev_current
        next_change = this_current - next_current
        if prev_change * next_change >= 0:
            extrema_change = np.abs(this_current - extrema[-1][1])
            if extrema_change > diff_threshold:
                extrema.append(event[i])
    
    extrema.append(event[-1])

    return np.array(extrema)

def find_current_diffs(event):
    current_diffs = []

    for i in range(1, len(event)):
        current_diffs.append(event[i, 1] - event[i-1, 1])
    
    return np.array(current_diffs)

def plot_data(data, plot_extrema=False, diff_threshold=2):
    k = len(data)
    fig, axes = plt.subplots(k, 1, figsize=(30, 15), sharex=True, sharey=True)

    for i in range(k):
        event = data[i]
        time = event[:, 0]
        current = event[:, 1]

        g = sns.lineplot(x=time, y=current, ax=axes[i])
        g.set_xlabel('Time [ms]')
        g.set_ylabel('Current')

        if plot_extrema:
            # extrema = find_extrema(event, diff_threshold=diff_threshold)
            peaks_idx = find_peaks_cwt(event[:, 1], widths=event[-1][0])
            peaks = event[peaks_idx]
            time = peaks[:, 0]
            current = peaks[:, 1]
            g = sns.scatterplot(x=time, y=current, ax=axes[i], s=100, color='blue')

def build_features(event, diff_threshold=0):
    features = {
        'num_signals': 0, 
        'duration': 0,
        'max_current': 0, 
        'min_current': 0, 
        'mean_current': 0,
        'std_current': 0,
        # 'num_extrema': 0, 
        # 'mean_extrema': 0,
        # 'std_extrema': 0,
        # 'mean_extrema_diff': 0,
        'num_peaks': 0,
        'mean_peaks': 0,
        'peak_1': 0,
        'peak_2': 0,
        'peak_3': 0,
        'peak_4': 0,
        'peak_5': 0
    }
    if len(event) > 0:
        features['num_signals'] = len(event)
        features['duration'] = event[-1][0]
        features['max_current'] = np.max(event[:, 1])
        features['min_current'] = np.min(event[:, 1])
        features['mean_current'] = np.mean(event[:, 1])
        features['std_current'] = np.std(event[:, 1])
        # extrema = find_extrema(event, diff_threshold=diff_threshold)
        # features['num_extrema'] = len(extrema)
        # features['mean_extrema'] = np.mean(extrema)
        # features['std_extrema'] = np.std(extrema)
        # features['mean_extrema_diff'] = np.mean(np.abs([extrema[i, 1] - extrema[i-1, 1] for i in range(1, len(extrema))]))
        peaks_idx = find_peaks_cwt(event[:, 1], widths=max([1, event[-1][0]]))
        features['num_peaks'] = len(peaks_idx)
        if len(peaks_idx) > 0:
            peaks = sorted(event[peaks_idx][:, 1], reverse=True)
            features['mean_peaks'] = np.mean(peaks)
            features['peak_1'] = peaks[0] if len(peaks) > 0 else 0
            features['peak_2'] = peaks[1] if len(peaks) > 1 else 0
            features['peak_3'] = peaks[2] if len(peaks) > 2 else 0
            features['peak_4'] = peaks[3] if len(peaks) > 3 else 0
            features['peak_5'] = peaks[4] if len(peaks) > 4 else 0
    return features