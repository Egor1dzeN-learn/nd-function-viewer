"""Base class for custom functions in Python.

Put a .py file into plugins/, derive a class from Function and restart the application.
The class is found automatically and appears in the Plugins tab, nothing has to be built.
A complete example is in plugins/example_functions.py.
"""
from abc import ABC, abstractmethod

import numpy as np


class Function(ABC):
    name: str | None = None  # shown in the list, must be unique; the class name by default
    min_dimension: int = 1   # the viewer supports dimensions 1 to 12
    max_dimension: int = 12

    @abstractmethod
    def bounds(self, dimension: int) -> list[tuple[float, float]]:
        """(lower, upper) of each variable. Above 4D all variables must share one range."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Returns n values for x of shape (n, dimension).

        x holds n points at once, up to millions of them, so work with numpy columns such as x[:, 0]
        instead of looping over the points.
        """

    def minimum(self, dimension: int) -> tuple[list[float], float] | None:
        """Override to mark the global minimum on the plot: return (point, value)."""
        return None


def box(dimension: int, lower: float, upper: float) -> list[tuple[float, float]]:
    """The same range for all variables."""
    return [(lower, upper)] * dimension
