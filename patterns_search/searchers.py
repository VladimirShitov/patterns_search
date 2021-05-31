from pathlib import Path
import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .augmentations import (
    Augmentation,
    AugmentationParameter,
    drop_random_points,
    insert_points,
    stretch_function,
    remove_random_regions,
)
from .utils import (
    apply_cross_correlation,
    fill_with_the_best_score,
    find_distant_dots,
    make_vertical_pattern_target_plot,
    scale_curve,
)

DEFAULT_AUGMENTATIONS = (
    Augmentation(
        fun=drop_random_points,
        fun_parameters=[
            AugmentationParameter(
                name="keep_probability",
                generating_function=np.random.uniform,
                generating_parameters={"low": 0.1, "high": 0.5},
            ),
        ]
    ),
    Augmentation(
        fun=remove_random_regions,
        fun_parameters=[
            AugmentationParameter(
                name="fraction",
                generating_function=np.random.randint,
                generating_parameters={"low": 10, "high": 20}
            ),
            AugmentationParameter(
                name="n_regions",
                generating_function=np.random.randint,
                generating_parameters={"low": 2, "high": 4}
            ),
        ]
    ),
    Augmentation(
        fun=insert_points,
        fun_parameters=[
            AugmentationParameter(
                name="insertion_probability",
                generating_function=np.random.uniform,
                generating_parameters={"low": 0.3, "high": 1},
            ),
        ]
    ),
    Augmentation(
        fun=stretch_function,
        fun_parameters=[
            AugmentationParameter(
                name="stretching_coef",
                generating_function=np.random.uniform,
                generating_parameters={"low": 3, "high": 10},
            ),
        ]
    ),
)


class PatternSearcher:
    def __init__(
        self,
        augmentations: Iterable[Augmentation] = DEFAULT_AUGMENTATIONS,
        n_augmentations: int = 10,
        scale_target: bool = True,
        pattern_name: str = ""
    ):
        self.pattern = None
        self.augmented_patterns = []
        self.pattern_name = pattern_name
        self.augmentations = augmentations
        self.n_augmentations = n_augmentations
        self.scale_target = scale_target

    def fit(self, pattern):
        self.pattern = pattern
        self.augmented_patterns.append(self.pattern)

        for _ in range(self.n_augmentations):
            augmentation: Augmentation = random.choice(self.augmentations)
            augmented_pattern = augmentation.generate(self.pattern)
            self.augmented_patterns.append(augmented_pattern)

    def predict(self, target: np.array):
        if self.scale_target:
            target = scale_curve(target)

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
        while len(new_pattern) > len(target):
            indexes = np.random.randint(0, len(new_pattern), size=len(new_pattern)-len(target))
            new_pattern = np.delete(new_pattern, indexes)

        return new_pattern

    def visualise_history(self, cut_of_point: float = 0.9, save_dir: Path = None, show_plots=True):
        for i, search_result in enumerate(self.history):
            n_good_results = len(search_result.result[search_result.result >= cut_of_point])
            if not n_good_results:
                continue

            top_results = search_result.result.argsort()[::-1]
            best_hits = find_distant_dots(
                top_results,
                result_length=min(3, n_good_results),
                min_distance=len(search_result.target) // 5
            )

            title = f"{self.pattern_name} in {search_result.target.name}"

            fig, axes = make_vertical_pattern_target_plot(
                pattern=search_result.pattern,
                target=search_result.target,
                labels=search_result.target.depth,
                starts=best_hits,
                offset=len(search_result.target) // 3,
                pattern_name=title,
                starts_confidence=search_result.result[best_hits]
            )
            if save_dir:
                fig.savefig(save_dir / f"{search_result.target.name}_{self.pattern_name}_result_{i}.png")
            if show_plots:
                fig.show()

            plt.close(fig)


class CorrelationSearcher(PatternSearcher):
    def __init__(
        self,
        scale_target: bool = False,
        augmentations: Iterable[Augmentation] = DEFAULT_AUGMENTATIONS,
        n_augmentations: int = 10,
        pattern_name: str = ""
    ):
        super().__init__(
            augmentations=augmentations,
            n_augmentations=n_augmentations,
            scale_target=scale_target,
            pattern_name=pattern_name
        )

    def fit(self, pattern):
        super().fit(pattern)

    def predict(self, target: np.array):
        super().predict(target=target)

        results = np.zeros(len(target))

        for augmented_pattern in self.augmented_patterns:
            pattern = augmented_pattern.copy()

            if len(pattern) > len(target):
                pattern = self.make_pattern_shorter(pattern, target)

            assert len(pattern) <= len(target)

            result = apply_cross_correlation(target=target, pattern=pattern)

            results = fill_with_the_best_score(
                previous_scores=results, scores=result, pattern_length=len(pattern)
            )

        return results
