from pathlib import Path
from typing import List, Optional, Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_curve(curve: np.array) -> np.array:
    """Scale `curve` from 0 to 1 and subtract it from 1"""
    scaler = MinMaxScaler()
    new_curve = scaler.fit_transform(curve.reshape(-1, 1)).flatten()
    return 1 - new_curve


def find_distant_dots(dots: np.array, result_length: int, min_distance: int):
    """Filter `dots` so that minimum distance between them is more or equals than `min_distance`

    Parameters
    ----------
    dots: np.array or List[Number]
        Array with points that you want to filter
    result_length: int
        How many dots to include in final result
    min_distance: int
        If distance between any dots is less than `min_distance` one of the dots will be removed

    Returns
    -------
    List of filtered dots with maximum length equals to `result_length`. It is guaranteed that the distance
        between any pair of dots is more or equals than `min_distance`

    Examples
    --------
    >>> arr = [1, 5, 7, 3, 10]
    >>> find_distant_dots(arr, result_length=3, min_distance=3)
    [1, 5, 10]
    >>> find_distant_dots(arr, result_length=3, min_distance=2)
    [1, 5, 7]
    """

    result = []

    for dot in dots:
        if all(abs(dot - x) >= min_distance for x in result):
            result.append(dot)
            if len(result) == result_length:
                return result

    return result


def apply_sliding_function(
    sequence_1,
    sequence_2,
    fun,
    do_center: bool = False,
    plot: bool = False,
    *args,
    **kwargs,
):
    """Apply `fun` to `sequence_1` and `sequence_2`. If one sequence is shorter, apply it with a sliding window

    Parametets
    ----------
    sequence_1: np.array-like with numbers
    sequence_2: np.array-like with numbers
    fun: Callable[[np.array, np.array], List[float]]
        Function to apply to sequences
    do_center: bool = False
        If set to True, sequences will be centered to have 0 mean
    plot: bool = False
        If set to True, plot some graphs

    Returns
    -------
    np.array with results of applying `fun` to `sequence_1` and slices of `sequence_2`
    """
    if len(sequence_1) < len(sequence_2):
        warnings.warn(
            "Length of sequence_1 is less than length of sequence_2. Swapping sequences"
        )
        sequence_1, sequence_2 = sequence_2, sequence_1

    n = (
        len(sequence_1) - len(sequence_2) + 1
    )  # How many times to slide with `sequence_2` through `sequence_1`
    result = np.zeros(n)
    window_size = len(sequence_2)

    if do_center:
        sequence_2 = sequence_2 - np.mean(sequence_2)
        if plot:
            plt.plot(np.arange(window_size), sequence_2)

    normalizing_coef = 1 / fun(sequence_2, sequence_2).item()

    for i in range(n):
        seq_1_slice = sequence_1[i : i + window_size].copy()
        if do_center:
            seq_1_slice = seq_1_slice - np.mean(seq_1_slice)
            if plot:
                plt.plot(np.arange(window_size), seq_1_slice)
        result[i] = (
            fun(seq_1_slice, sequence_2, *args, **kwargs).item() * normalizing_coef
        )

    return result


def make_vertical_pattern_target_plot(
    pattern: np.array,
    target: np.array,
    starts: List[int],
    labels: Optional[Iterable] = None,
    offset: int = 0,
    scale: bool = False,
    pattern_name: str = "",
    save_path: Optional[Path] = None,
    starts_confidence=None,
):
    """Plot pattern and the target function

    Parameters
    ----------
    pattern: np.array[Number]
        np.array with pattern you've tried to find in `target`
    target: np.array[Number]
        np.array, which represents the target function
    starts: List[int]
        Lists with indexes, which indicate where `pattern` starts in `target`
    labels: list
        List-like object with labels for `target`. It is used for ticks of the plot
    offset: int = 0
        How many data points to show around `pattern` in `target`
    scale: bool = False
        If True, both `pattern` and `target` will be scaled to [0; 1]
    pattern_name: str
        Title at the top of the plot
    save_path: Path
        Path where to save a plot
    starts_confidence: np.array[int]
        List of confidence scores for `starts`. Will be displayed as a title of plot
    """

    fig, axes = plt.subplots(
        nrows=1, ncols=len(starts) + 1, figsize=(4 + 4 * len(starts), 8), dpi=300
    )
    pattern_coordinates = np.arange(len(pattern))

    axes[0].plot(pattern, pattern_coordinates, "r", label="pattern")
    axes[0].legend(loc="upper right")
    axes[0].invert_xaxis()
    axes[0].invert_yaxis()
    top, bottom = axes[0].get_ylim()
    axes[0].set_ylim(top + offset, bottom - offset)

    for i, start in enumerate(starts, 1):

        start_coordinate = max(0, start - offset)
        end_coordinate = min(len(target), start + len(pattern) + offset)

        function_slice = target[start_coordinate:end_coordinate]

        if scale:
            scaler = MinMaxScaler()
            function_slice = scaler.fit_transform(function_slice.reshape(-1, 1))
            pattern = scaler.fit_transform(pattern)

        if labels is not None:
            depth_labels = labels[start_coordinate:end_coordinate]
        else:
            depth_labels = np.arange(len(function_slice))

        axes[i].plot(function_slice, depth_labels, color="blue")

        # Plot pattern as a red line
        if offset:
            bottom_line_pos = min(start + len(pattern), len(labels) - 1)
            axes[i].plot(
                target[start:bottom_line_pos],
                labels[start:bottom_line_pos],
                color="red",
                label=f"Top {i} result",
            )

        axes[i].legend(loc="upper right")

        axes[i].invert_xaxis()
        axes[i].invert_yaxis()
        if starts_confidence is not None:
            confidence = round(starts_confidence[i - 1], 2)
            axes[i].set_title(f"Confidence: {confidence}")

    if pattern_name:
        fig.suptitle(pattern_name, fontsize=16)

    if save_path:
        plt.savefig(save_path)

    return axes
