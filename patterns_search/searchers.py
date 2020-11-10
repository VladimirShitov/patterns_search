import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .augmentations import Augmentation, drop_random_points
from .types import SearchResult
from .utils import (
    apply_sliding_function,
    find_distant_dots,
    make_vertical_pattern_target_plot,
    scale_curve,
)


class PatternSearcher:
    def __init__(
        self,
        pattern: np.array,
        targets: List[np.array],
        augmentations: List[Augmentation],
        scale_target: bool = True,
    ):
        self.pattern = pattern
        self.targets = targets
        self.augmentations = augmentations
        self.history: List[SearchResult] = []

        if scale_target:
            self.targets = [scale_curve(target) for target in self.targets]

    @staticmethod
    def make_pattern_shorter(
        pattern: np.array, target: np.array, ratio: float = 1.0
    ) -> np.array:
        """Remove elements of `pattern` to make it shorter than the `target`

        Parameters
        ----------
        pattern: np.array
            Array, which you want to search for in the `target`
        target: np.array
            Array with the target function. You probably want `pattern` to have a shorter length than it
        ratio: float
            Ratio between new length of the `pattern` and length of the `target`. E. g. if `ratio` == 1, then
            new length of the `pattern` will be exactly the same as the length of the `target`. If `ratio` == 0.5,
            then new length of the `pattern` will be 2 times less than the length of the `target`, etc.

        Returns
        -------
        new_pattern: np.array
            `pattern` with some of the points removed
        """
        dropping_probability = len(target) * ratio / len(pattern)

        new_pattern = drop_random_points(pattern, keep_probability=1-dropping_probability)

        # Remove some points if pattern is still longer that the target
        if len(new_pattern) > len(target):
            indexes = np.random.randint(0, len(new_pattern), size=len(new_pattern)-len(target))
            new_pattern = np.delete(new_pattern, indexes)

        return new_pattern

    def visualise_history(self, cut_of_point: float = 0.9):
        for search_result in self.history:
            n_good_results = len(search_result.result[search_result.result >= cut_of_point])
            if not n_good_results:
                continue

            top_results = search_result.result.argsort()[::-1]
            best_hits = find_distant_dots(
                top_results,
                result_length=min(3, n_good_results),
                min_distance=len(search_result.target) // 5
            )

            make_vertical_pattern_target_plot(
                pattern=search_result.pattern,
                target=search_result.target,
                labels=np.arange(len(search_result.target)),
                starts=best_hits,
                offset=len(search_result.target) // 3,
                pattern_name="Pattern",
                starts_confidence=search_result.result[best_hits]
            )
            plt.show()

    def search(self):
        raise NotImplementedError


class CorrelationSearcher(PatternSearcher):
    def __init__(
        self,
        pattern: np.array,
        targets: List[np.array],
        augmentations: List[Augmentation],
    ):
        super().__init__(pattern, targets, augmentations)

    def search(self, n_tries: int = 1):
        for _ in range(n_tries):
            for target in self.targets:
                augmentation: Augmentation = random.choice(self.augmentations)
                augmented_pattern = augmentation.generate(self.pattern)

                if len(augmented_pattern) > len(target):
                    augmented_pattern = self.make_pattern_shorter(augmented_pattern, target)

                result = apply_sliding_function(
                    sequence_1=target,
                    sequence_2=augmented_pattern,
                    fun=np.correlate,
                    do_center=True,
                )

                self.history.append(SearchResult(pattern=augmented_pattern, target=target, result=result))
