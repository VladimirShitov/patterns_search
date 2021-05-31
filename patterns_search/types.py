from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    pattern: np.array
    target: Target
    result: np.array
