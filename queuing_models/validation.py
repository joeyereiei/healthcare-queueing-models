"""Input validation for the queuing models."""
import math

import numpy as np

from .exceptions import InvalidParameterError


def _is_real_number(value) -> bool:
    """Return True if ``value`` is a finite real number (booleans excluded)."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(value)
    )


def validate_params(lambd: float, mu: float, s: int = 1, cv: float = 1.0) -> None:
    """Check queue parameters and raise if they are invalid or unstable.

    Parameters
    ----------
    lambd : float
        Mean arrival rate (> 0).
    mu : float
        Mean service rate per server (> 0).
    s : int, default 1
        Number of servers (integer >= 1).
    cv : float, default 1.0
        Coefficient of variation of service time (>= 0, M/G/1 only).

    Raises
    ------
    InvalidParameterError
        If any value is out of range or lambd / (s * mu) >= 1.
    """
    if not (_is_real_number(lambd) and _is_real_number(mu)):
        raise InvalidParameterError(
            f"lambd and mu must be finite numbers (got {lambd!r}, {mu!r})."
        )
    if not (lambd > 0 and mu > 0):
        raise InvalidParameterError(f"lambd and mu must be > 0 (got {lambd}, {mu}).")
    if isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 1:
        raise InvalidParameterError(f"s must be an integer >= 1 (got {s!r}).")
    if not _is_real_number(cv) or cv < 0:
        raise InvalidParameterError(f"cv must be a finite number >= 0 (got {cv!r}).")
    rho = lambd / (s * mu)
    if rho >= 1:
        raise InvalidParameterError(
            f"rho = {rho:.3f} >= 1: no steady state. Lower lambd or add capacity."
        )
