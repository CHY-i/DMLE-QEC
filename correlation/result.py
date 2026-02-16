"""The result of correlation analysis."""
import dataclasses
from typing import Union, Tuple, Dict, Iterable, Optional

import numpy as np

from correlation.utils import HyperEdge


AnalyticalResult = Tuple[np.ndarray, np.ndarray]

NumericalResult = Dict[HyperEdge, float]


@dataclasses.dataclass(frozen=True)
class CorrelationResult:
    """Dataclass for storing the result of correlation analysis."""
    data: Union[AnalyticalResult, NumericalResult]

    @property
    def highest_order(self) -> int:
        """Return the highest order of correlation analysis."""
        if isinstance(self.data, dict):
            return max(len(edge) for edge in self.data)
        return 2

    def get(self, detectors: Iterable[int]) -> Optional[float]:
        """Return the correlation of the given detectors.

        Args:
            detectors: The detectors to get the correlation of.

        Returns:
            The correlation of the given detectors. If the correlation is not
            included in the result, return None.
        """
        detectors = list(detectors)
        if isinstance(self.data, dict):
            hyperedge = frozenset(detectors)
            val = self.data.get(hyperedge)
        elif len(detectors) == 2:
            val = self.data[1][detectors[0], detectors[1]]
        elif len(detectors) == 1:
            val = self.data[0][detectors[0]]
        else:
            val = None
        return val
