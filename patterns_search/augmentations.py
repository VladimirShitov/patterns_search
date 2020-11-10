from typing import Callable, List

import numpy as np
from scipy.interpolate import interp1d


def drop_random_points(x: np.array, keep_probability=0.5) -> np.array:
    """Drop random points from array `x`"""
    keep_probabilities = np.random.random(size=len(x))
    return x[keep_probabilities < keep_probability]


def insert_points(x: np.array, insertion_probability=0.5) -> np.array:
    """Insert new points to array `x`

    Insertion probability after each point in `x` can be set with `insertion_probability`.
    Values of new points are interpolated from the closest points."""

    new_array = []

    for i in range(len(x)):
        new_array.append(x[i])

        if np.random.random() <= insertion_probability and i < len(x) - 1:
            new_array.append((x[i] + x[i + 1]) / 2)

    return np.array(new_array)


def stretch_function(x: np.array, stretching_coef: float):
    """Stretch `x` so that it's size will be `len(x)*stretching_coef`"""

    scope = np.arange(len(x))
    f = interp1d(scope, x)

    stretch_times = int(stretching_coef)
    new_points = np.linspace(0, scope[-1], len(scope) * stretch_times)
    interpolated_array = f(new_points)

    insertion_probability = stretching_coef % 1
    array_with_extra_points = insert_points(interpolated_array, insertion_probability)

    return array_with_extra_points


def remove_region(x: np.array, region_start: int, region_end: int) -> np.array:
    """Remove region from `region_start` to `region_end` from `x`"""
    new_array = x[:region_start].tolist()
    new_array.extend(x[region_end:].tolist())

    return np.array(new_array)


def remove_random_region(x: np.array, region_size: int) -> np.array:
    """Remove random region of length `region_size` from `x`"""

    dropping_start = int(np.random.uniform(0, len(x) - region_size))

    return remove_region(
        x, region_start=dropping_start, region_end=dropping_start + region_size
    )


def remove_random_regions(x: np.array, fraction: int, n_regions: int) -> np.array:
    """Remove random region of size `1/fraction` from `x` `n_regions` times"""

    new_array = remove_random_region(x, region_size=len(x) // fraction)

    for _ in range(n_regions - 1):
        new_array = remove_random_region(
            new_array, region_size=len(new_array) // fraction
        )

    return new_array


class AugmentationParameter:
    """Class for creating parameter, which value should be randomly generated

    Examples
    --------
    >>> some_parameter = AugmentationParameter(
    >>>     name='score', generating_function=np.random.randint generating_parameters={"low": 3, "high": 10},
    >>> )
    >>> some_parameter.generate()
    {'score': 4}
    >>> some_parameter.generate()
    {'score': 8}
    >>> for _ in range(3):
    >>>     augmentation = SomeAugmentation(**some_parameter.generate())  # Each augmentation will be different
    """

    def __init__(
        self, name: str, generating_function: Callable, generating_parameters: dict
    ):
        """Set `name` of the parameter and function, which will generate it's value

        Parameters
        ----------
        name: str
            Name of the parameter
        generating_function: Callable
            Function, which is used to generate parameter's value
        generating_parameters: dict
            Required parameters of `generating_function`
        """
        self.name = name
        self.generating_function = generating_function
        self.generating_parameters = generating_parameters

    def generate(self):
        """Return dictionary with name of the parameter as a key and random value as a dictionary value"""
        return {self.name: self.generating_function(**self.generating_parameters)}


class Augmentation:
    """Class that helps to create augmentation with specific function, and apply it with different parameters

    Examples
    --------
    >>> removing_regions_augmentation = Augmentation(
    >>>     fun=remove_random_regions,
    >>>     fun_parameters=[
    >>>         AugmentationParameter(
    >>>             name="fraction",
    >>>             generating_function=np.random.randint,
    >>>             generating_parameters={"low": 10, "high": 15},
    >>>         ),
    >>>         AugmentationParameter(
    >>>             name="n_regions",
    >>>             generating_function=np.random.randint,
    >>>             generating_parameters={"low": 1, "high": 3},
    >>>         ),
    >>>     ]
    >>> )
    >>> x = np.arange(15)
    >>> x
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
    >>> removing_regions_augmentation.generate(x)
    [ 0  1  2  3  4  5 13 14]
    """

    def __init__(self, fun, fun_parameters: List[AugmentationParameter]):
        self.fun = fun
        self.fun_parameters = fun_parameters

    def generate(self, array):
        """Generate augmented array"""
        current_parameters = {}
        for p in self.fun_parameters:
            current_parameters.update(p.generate())

        return self.fun(array, **current_parameters)
