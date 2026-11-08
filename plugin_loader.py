"""Finds the functions shown in the Plugins tab.

Python plugins are the classes derived from nd_function.Function in plugins/*.py, C++ plugins are
in native/plugins.dll. Files and classes whose names start with "_" are skipped.
A broken plugin is reported and does not affect the others.
"""
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

import test_functions as tf
from function_parser import MAX_DIMENSION, VARIABLES
from nd_function import Function

PLUGINS_DIR = Path(__file__).resolve().parent / "plugins"


class PythonProblem:
    """A Python plugin of a given dimension, with the interface of test_functions.TestProblem."""

    def __init__(self, function: Function, title: str, dimension: int):
        self._function = function
        self.title = title
        self.dimension = dimension
        if not function.min_dimension <= dimension <= function.max_dimension:
            raise ValueError(f"{title}: dimension {dimension} is not supported")

        bounds = self._guarded(lambda: np.asarray(function.bounds(dimension), dtype=float))
        if bounds.shape != (dimension, 2):
            raise ValueError(f"{title}: bounds() must return (lower, upper) for each variable")
        self.lower, self.upper = bounds[:, 0], bounds[:, 1]
        if not np.all(self.lower < self.upper):
            raise ValueError(f"{title}: a lower bound must be less than the upper bound")
        if dimension > len(VARIABLES) and (np.ptp(self.lower) > 0 or np.ptp(self.upper) > 0):
            raise ValueError(f"{title}: above {len(VARIABLES)}D all variables must share one range")

        self.optimum_point, self.optimum_value = None, None
        minimum = self._guarded(lambda: function.minimum(dimension))
        if minimum is not None:
            point, value = self._guarded(lambda: (np.asarray(minimum[0], dtype=float), float(minimum[1])))
            if point.shape != (dimension,):
                raise ValueError(f"{title}: minimum() must give one coordinate per variable")
            if not np.all((self.lower <= point) & (point <= self.upper)):
                raise ValueError(f"{title}: the minimum lies outside the bounds")
            self.optimum_point, self.optimum_value = point, value

    def _guarded(self, body: Callable):
        """Errors in plugin code become ValueError, which the application shows in the status bar."""
        try:
            with np.errstate(all="ignore"):
                return body()
        except Exception as exc:
            raise ValueError(f"{self.title}: {type(exc).__name__}: {exc}") from exc

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """points is an array of shape (n, dimension); returns n function values."""
        points = np.asarray(points, dtype=float).reshape(-1, self.dimension)
        values = self._guarded(lambda: np.asarray(self._function.evaluate(points), dtype=float))
        if values.ndim == 0:  # a constant
            return np.full(len(points), float(values))
        if values.shape != (len(points),):
            raise ValueError(
                f"{self.title}: evaluate() returned shape {values.shape} for {len(points)} points. "
                "x has shape (n, dimension): use columns such as x[:, 0]"
            )
        return values


@dataclass(frozen=True)
class PluginInfo:
    name: str
    dimensions: tuple
    create: Callable  # dimension -> problem


def _import_file(path: Path):
    name = f"nfv_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_info(cls: type) -> PluginInfo:
    function = cls()
    name = str(function.name or cls.__name__)
    dimensions = tuple(range(max(function.min_dimension, 1), min(function.max_dimension, MAX_DIMENSION) + 1))
    if not dimensions:
        raise ValueError(f"supports no dimension from 1 to {MAX_DIMENSION}")
    return PluginInfo(name, dimensions, partial(PythonProblem, function, name))


def _python_plugins(errors: list) -> list:
    plugins = []
    for path in sorted(PLUGINS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _import_file(path)
        except (Exception, SystemExit) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or not issubclass(cls, Function) or class_name.startswith("_"):
                continue
            if inspect.isabstract(cls):
                errors.append(f"{path.name}: {class_name} does not implement {', '.join(sorted(cls.__abstractmethods__))}")
                continue
            try:
                plugins.append(_make_info(cls))
            except Exception as exc:
                errors.append(f"{path.name}: {class_name}: {type(exc).__name__}: {exc}")
    return plugins


def _cpp_plugins(errors: list) -> list:
    if not tf.PLUGIN_DLL_PATH.exists():
        return []
    try:
        catalog = tf.plugin_catalog()
    except RuntimeError as exc:
        errors.append(f"{tf.PLUGIN_DLL_PATH.name}: {exc}")
        return []
    return [PluginInfo(info.name, info.dimensions, partial(tf.plugin_problem, info.name)) for info in catalog]


def load_catalog() -> tuple[list[PluginInfo], list[str]]:
    """Returns the plugins sorted by name and the messages about those that could not be loaded."""
    errors = []
    catalog = {}
    for info in [*_python_plugins(errors), *_cpp_plugins(errors)]:
        if info.name in catalog:
            errors.append(f"Several plugins use the same name, only the first is shown: {info.name}")
        else:
            catalog[info.name] = info
    return sorted(catalog.values(), key=lambda info: info.name), errors
