"""Example of custom functions. Copy this file, change the classes and restart the application."""
import numpy as np

from nd_function import Function, box


class ShiftedBowl(Function):
    """Any dimension from 1 to 12, the global minimum is known and marked on the plot."""

    name = "Python example: shifted bowl"

    def bounds(self, dimension):
        return box(dimension, -5, 5)

    def evaluate(self, x):
        return np.sum((x - 1) ** 2, axis=1)

    def minimum(self, dimension):
        return [1.0] * dimension, 0.0


class Himmelblau(Function):
    """Only 2D, each variable has its own range, the minimum is not specified."""

    name = "Python example: Himmelblau"
    min_dimension = 2
    max_dimension = 2

    def bounds(self, dimension):
        return [(-5, 5), (-6, 6)]

    def evaluate(self, x):
        a, b = x[:, 0], x[:, 1]
        return (a**2 + b - 11) ** 2 + (a + b**2 - 7) ** 2
