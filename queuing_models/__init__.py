"""Analytic queuing models (M/M/1, M/G/1, M/M/s) for healthcare capacity planning."""
from .exceptions import InvalidModelError, InvalidParameterError, MissingColumnsError
from .formulas import (
    compute_Lq, compute_Ls, compute_metrics, compute_p, compute_p0, compute_rho,
    compute_Wq, compute_Ws,
)
from .queue import Queue, find_parameter
from .validation import validate_params

__all__ = [
    "Queue", "find_parameter", "validate_params", "compute_metrics",
    "compute_rho", "compute_p0", "compute_Lq", "compute_Ls", "compute_Wq",
    "compute_Ws", "compute_p",
    "InvalidModelError", "InvalidParameterError", "MissingColumnsError",
]
__version__ = "1.0.0"
