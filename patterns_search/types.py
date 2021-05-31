from dataclasses import dataclass

import numpy as np


@dataclass
class Target:
    sp: np.array
    depth: np.array
    name: str


@dataclass
class SearchResult:
    pattern: np.array
    target: Target
    result: np.array
