"""Global optimization test functions from the global-opt project.

The functions are evaluated in native/testfuncs.dll (see native/build.py), built from the
GCGen and global-opt classic function sources, so they match them bit for bit.
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from function_parser import MAX_DIMENSION

DLL_PATH = Path(__file__).resolve().parent / "native" / "testfuncs.dll"

_lib = None
_load_error = None


def _library() -> ctypes.CDLL:
    global _lib, _load_error
    if _lib is not None:
        return _lib
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        lib = ctypes.CDLL(str(DLL_PATH))
    except OSError as exc:
        _load_error = f"Failed to load {DLL_PATH.name}: {exc}. Build it with: python native/build.py"
        raise RuntimeError(_load_error) from None

    vp, dp, ip = ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int
    lib.tf_last_error.restype = ctypes.c_char_p
    lib.tf_create_gcgen.argtypes, lib.tf_create_gcgen.restype = [ip, ip, ip, ip, ip], vp
    lib.tf_create_classic.argtypes, lib.tf_create_classic.restype = [ctypes.c_char_p, ip], vp
    lib.tf_destroy.argtypes = [vp]
    lib.tf_dimension.argtypes, lib.tf_dimension.restype = [vp], ip
    lib.tf_bounds.argtypes = [vp, dp, dp]
    lib.tf_optimum.argtypes, lib.tf_optimum.restype = [vp, dp, dp], ip
    lib.tf_evaluate.argtypes, lib.tf_evaluate.restype = [vp, dp, ip, dp], ip
    lib.tf_classic_names.argtypes, lib.tf_classic_names.restype = [ctypes.c_char_p, ip], ip
    _lib = lib
    return lib


def is_available() -> tuple[bool, str]:
    try:
        _library()
        return True, ""
    except RuntimeError as exc:
        return False, str(exc)


def _as_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class TestProblem:
    """A single test problem: dimension, bounds, known minimum and evaluation."""

    def __init__(self, handle: int | None, title: str):
        lib = _library()
        if not handle:
            raise ValueError(lib.tf_last_error().decode(errors="replace"))
        self._handle = handle
        self.title = title
        self.dimension = lib.tf_dimension(handle)

        self.lower = np.empty(self.dimension)
        self.upper = np.empty(self.dimension)
        lib.tf_bounds(handle, _as_ptr(self.lower), _as_ptr(self.upper))

        point, value = np.empty(self.dimension), np.empty(1)
        known = lib.tf_optimum(handle, _as_ptr(point), _as_ptr(value))
        self.optimum_point = point if known else None
        self.optimum_value = float(value[0]) if known else None

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """points is an array of shape (n, dimension); returns n function values."""
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, self.dimension)
        out = np.empty(len(points))
        if not _library().tf_evaluate(self._handle, _as_ptr(points), len(points), _as_ptr(out)):
            raise ValueError(_library().tf_last_error().decode(errors="replace"))
        return out

    def __del__(self):
        if getattr(self, "_handle", None) and _lib is not None:
            _lib.tf_destroy(self._handle)
            self._handle = None


@dataclass(frozen=True)
class Family:
    name: str
    code: int          # family code passed to tf_create_gcgen
    dimensions: tuple  # supported dimensions
    count: int         # problems are numbered 1..count in the UI
    first_index: int   # index of the first problem in GCGen (0 or 1)


GCGEN_FAMILIES = [
    Family("Hill", 0, (1,), 1000, 0),
    Family("Shekel", 1, (1,), 1000, 0),
    Family("Hansen", 2, (1,), 20, 0),
    Family("Grishagin", 3, (2,), 100, 1),
    Family("GKLS", 4, tuple(range(2, MAX_DIMENSION + 1)), 100, 1),
]
GKLS_CLASSES = ["Simple", "Hard"]   # → GKLSClass: Simple = 1, Hard = 0
GKLS_TYPES = ["D", "ND", "D2"]      # → GKLSFuncionType: ND = 0, D = 1, D2 = 2
_GKLS_CLASS_CODES = {"Hard": 0, "Simple": 1}
_GKLS_TYPE_CODES = {"ND": 0, "D": 1, "D2": 2}


def gcgen_problem(
    family: Family, number: int, gkls_class: str = "Simple", gkls_type: str = "D", dimension: int | None = None
) -> TestProblem:
    lib = _library()
    dimension = dimension if dimension in family.dimensions else family.dimensions[0]
    handle = lib.tf_create_gcgen(
        family.code, number - 1 + family.first_index, dimension,
        _GKLS_CLASS_CODES[gkls_class], _GKLS_TYPE_CODES[gkls_type],
    )
    title = f"{family.name} #{number}"
    if family.name == "GKLS":
        title += f" ({dimension}D, {gkls_class}, {gkls_type})"
    return TestProblem(handle, title)


def classic_problem(name: str, dimension: int = 2) -> TestProblem:
    return TestProblem(_library().tf_create_classic(name.encode(), dimension), name)


@dataclass(frozen=True)
class ClassicInfo:
    name: str
    scalable: bool       # defined for any dimension
    dimensions: tuple    # which of dimensions 1..MAX_DIMENSION are supported


def classic_catalog() -> list[ClassicInfo]:
    """Classic functions that can be plotted (dimensions 1 to MAX_DIMENSION)."""
    lib = _library()
    size = lib.tf_classic_names(None, 0)
    buffer = ctypes.create_string_buffer(size)
    lib.tf_classic_names(buffer, size)

    catalog = []
    for name in buffer.value.decode().split():
        dims = []
        for dim in range(1, MAX_DIMENSION + 2):
            try:
                problem = classic_problem(name, dim)
            except ValueError:
                continue
            if problem.dimension != dim:
                continue
            # Some functions (e.g. rosenbrock) degenerate to a constant in 1D; skip those.
            samples = np.linspace(problem.lower, problem.upper, 7)
            if np.ptp(problem.evaluate(samples)) > 0:
                dims.append(dim)
        scalable = MAX_DIMENSION + 1 in dims
        drawable = tuple(d for d in dims if d <= MAX_DIMENSION)
        if drawable:
            catalog.append(ClassicInfo(name, scalable, drawable))
    return sorted(catalog, key=lambda info: (not info.scalable, info.name))
