"""Safe parsing and evaluation of formulas of many variables: x, y, z, w or x1, x2, ...

The formula is parsed into an AST and checked against a whitelist of nodes, names
and functions, so it cannot be used to run arbitrary Python code.
"""
import ast

import numpy as np

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "min": np.minimum, "max": np.maximum, "where": np.where,
}
CONSTANTS = {"pi": np.pi, "e": np.e}
MAX_DIMENSION = 12
VARIABLES = ("x", "y", "z", "w")
INDEXED_VARIABLES = tuple(f"x{i}" for i in range(1, MAX_DIMENSION + 1))  # x1 is x, x2 is y, x3 is z, x4 is w

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)

HELP_TEXT = (
    f"Variables: x, y, z, w or x1 … x{MAX_DIMENSION} (x1 is x, x2 is y, ...). Constants: pi, e.\n"
    "Operators: + - * / ^ (or **), comparisons (x > 0).\n"
    "Functions: " + ", ".join(FUNCTIONS) + "."
)


def variable_names(dimension: int) -> tuple:
    """Names shown for a function of this dimension."""
    return VARIABLES[:dimension] if dimension <= len(VARIABLES) else INDEXED_VARIABLES[:dimension]


def _position(name: str) -> int:
    return VARIABLES.index(name) if name in VARIABLES else INDEXED_VARIABLES.index(name)


class FormulaError(ValueError):
    """Error in the formula text or during its evaluation."""


class Formula:
    def __init__(self, text: str):
        source = text.strip().replace("^", "**")
        if not source:
            raise FormulaError("Enter a formula")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(f"Syntax error: {exc.msg}") from None

        names = set()
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise FormulaError(f"Construct not allowed: {type(node).__name__}")
            if isinstance(node, ast.Constant) and (
                isinstance(node.value, bool) or not isinstance(node.value, (int, float))
            ):
                raise FormulaError(f"Constant not allowed: {node.value!r}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                    raise FormulaError("Only functions from the list can be called")
                if node.keywords:
                    raise FormulaError("Keyword arguments are not supported")
            if isinstance(node, ast.Name):
                if node.id not in FUNCTIONS and node.id not in CONSTANTS and node.id not in (*VARIABLES, *INDEXED_VARIABLES):
                    raise FormulaError(f"Unknown name: {node.id}")
                names.add(node.id)

        self.text = text.strip()
        # A formula in x and z is treated as 3D: y simply does not affect the value.
        self.dimension = max((_position(v) + 1 for v in names if v in (*VARIABLES, *INDEXED_VARIABLES)), default=1)
        self._code = compile(tree, "<formula>", "eval")

    def evaluate(self, *axes: np.ndarray) -> np.ndarray:
        """Evaluates the formula on `dimension` arrays. The result has the shape of the first one."""
        namespace = {**FUNCTIONS, **CONSTANTS, **dict(zip(VARIABLES, axes)), **dict(zip(INDEXED_VARIABLES, axes))}
        try:
            with np.errstate(all="ignore"):
                result = eval(self._code, {"__builtins__": {}}, namespace)
            return np.broadcast_to(np.asarray(result, dtype=float), np.shape(axes[0])).copy()
        except Exception as exc:
            raise FormulaError(f"Evaluation error: {exc}") from None
