"""Reusable ECG preprocessing and evaluation helpers."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from sklearn.metrics import auc, confusion_matrix


def collapse_windows_to_centers(binary_vec, mode="center"):
    binary_vec = np.asarray(binary_vec, dtype=int)
    output = np.zeros_like(binary_vec)
    changes = np.diff(np.pad(binary_vec, (1, 1)))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    for start, end in zip(starts, ends):
        if mode == "center":
            index = (start + end) // 2
        elif mode == "left":
            index = start
        elif mode == "right":
            index = end
        else:
            raise ValueError("mode must be 'center', 'left', or 'right'")
        output[index] = 1
    return output


def add_tolerance(binary_vec, window):
    kernel = np.ones(2 * window + 1, dtype=int)
    return (np.convolve(binary_vec, kernel, mode="same") > 0).astype(int)


def coverage_pr_auc(
    score_signal,
    ground_truth,
    fs=None,
    thresholds=None,
    tolerance_window=1,
    check_sign=True,
):
    """Compute tolerance-aware precision, recall, and PR-AUC."""
    del fs
    score_signal = np.asarray(score_signal)
    ground_truth = np.asarray(ground_truth)
    if check_sign and abs(score_signal.min()) > abs(score_signal.max()):
        score_signal = -score_signal
    if thresholds is None:
        thresholds = np.linspace(score_signal.min(), score_signal.max() - 1e-5, 200)

    precision_values = []
    recall_values = []
    for threshold in thresholds:
        estimate = score_signal > threshold
        _, false_positive, _, true_positive = confusion_matrix(
            add_tolerance(ground_truth, tolerance_window), estimate, labels=[0, 1]
        ).ravel()
        _, false_negative, _, _ = confusion_matrix(
            add_tolerance(estimate, tolerance_window), ground_truth, labels=[0, 1]
        ).ravel()
        precision_values.append(
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0
        )
        recall_values.append(
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0
        )

    precision_values = np.asarray(precision_values)
    recall_values = np.asarray(recall_values)
    order = np.argsort(recall_values)
    return precision_values, recall_values, auc(
        recall_values[order], precision_values[order]
    )


coverage_PR_AUC = coverage_pr_auc


def lowpass_filter(signal, fs=1000, cutoff=100, order=5):
    coefficients = butter(order, cutoff / (0.5 * fs), btype="lowpass")
    return filtfilt(*coefficients, signal)


def remove_baseline_median(signal, window_size=101):
    return signal - median_filter(signal, size=window_size)


def preprocess_ecg(signal, fs=1000):
    filtered = lowpass_filter(signal, fs, cutoff=100)
    centered = remove_baseline_median(filtered, window_size=101)
    return (centered - np.mean(centered)) / np.std(centered)


def window_vector_to_signal(values, signal_length, lag, jump, fill_value=0.0):
    starts = np.arange(0, signal_length - lag, jump)
    output = np.full(signal_length, fill_value, dtype=float)
    for start, value in zip(starts, values):
        center = start + lag // 2
        if center < signal_length:
            output[center] = value
    return output


def const_lag(signal, lag, jump):
    signal_length = len(signal)
    lagged = np.column_stack([np.roll(signal, -offset) for offset in range(lag + 1)])
    return lagged[: signal_length - lag : jump]


def const_lags(signal, win_size=12, step=6):
    return sliding_window_view(signal, win_size)[::step]
