from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    pattern: np.array
    result: np.array
