"""The Queue class: a reusable interface for running queuing experiments."""
from typing import Callable

import numpy as np
import pandas as pd

from .exceptions import InvalidParameterError, MissingColumnsError
from .formulas import _check_model, compute_metrics

REQUIRED_COLUMNS = {
    "mm1": ["lambd", "mu"],
    "mg1": ["lambd", "mu", "cv"],
    "mms": ["lambd", "mu", "s"],
}

MODEL_LABELS = {"mm1": "M/M/1", "mg1": "M/G/1", "mms": "M/M/s"}


class Queue:
    """A queuing experiment for one model.

    Parameters
    ----------
    model : str
        ``"mm1"``, ``"mg1"`` or ``"mms"`` (case-insensitive).

    Raises
    ------
    InvalidModelError
        If the model name is unknown.

    Examples
    --------
    >>> round(Queue("mm1").experiment(lambd=2, mu=3).results["Ls"], 6)
    2.0
    """

    def __init__(self, model: str):
        self._model = _check_model(model)
        self._params: dict = {}
        self._results: dict = {}
        self._result_df: pd.DataFrame | None = None

    @property
    def model(self) -> str:
        """The model name."""
        return self._model

    def experiment(self, lambd: float, mu: float, s: int = 1, cv: float = 1.0,
                   n: int = 10) -> "Queue":
        """Run the model for one set of parameters.

        Parameters
        ----------
        lambd : float
            Mean arrival rate.
        mu : float
            Mean service rate per server.
        s : int, default 1
            Number of servers (M/M/s only).
        cv : float, default 1.0
            Coefficient of variation of service time (M/G/1 only).
        n : int, default 10
            Number of state probabilities p0..p_{n-1} to compute.

        Returns
        -------
        Queue
            This instance, so calls can be chained, e.g.
            ``Queue("mm1").experiment(2, 3).report()``.

        Raises
        ------
        InvalidParameterError
            If the parameters are invalid or give no steady state.
        """
        self._results = compute_metrics(self._model, lambd, mu, s, cv, n)
        self._params = {"lambd": lambd, "mu": mu, "s": s, "cv": cv}
        return self

    @property
    def results(self) -> dict:
        """Performance measures from the last call to ``experiment``."""
        return dict(self._results)

    def report(self, n: int = 10, digits: int = 5) -> None:
        """Print the results of the last call to ``experiment``.

        Parameters
        ----------
        n : int, default 10
            Maximum number of state probabilities to show.
        digits : int, default 5
            Number of decimal places.
        """
        if not self._results:
            raise RuntimeError("Run experiment() before report().")
        lines = [f"The result of the experiment ({MODEL_LABELS[self._model]}):", ""]
        params = {"lambda": self._params["lambd"], "mu": self._params["mu"]}
        if self._model == "mms":
            params["s"] = self._params["s"]
        if self._model == "mg1":
            params["CV"] = self._params["cv"]
        lines += [f"{k} = {v:.{digits}f}" if isinstance(v, float) else f"{k} = {v}"
                  for k, v in params.items()]
        lines += [f"{k} = {self._results[k]:.{digits}f}"
                  for k in ["rho", "Lq", "Ls", "Wq", "Ws"]]
        probs = [round(v, digits) for k, v in self._results.items() if k.startswith("p")]
        lines.append(f"p0, p1, ..., pn = {probs[:n]}" if len(probs) > 1 else
                     f"p0 = {probs[0]:.{digits}f}")
        print("\n".join(lines))

    def experiment_df(self, df: pd.DataFrame, n: int = 10,
                      on_unstable: str = "raise") -> pd.DataFrame:
        """Run the model for every row of a DataFrame of parameters.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per scenario, with columns ``lambd`` and ``mu`` (all models),
            ``cv`` (M/G/1) and ``s`` (M/M/s). Other columns are kept as labels.
        n : int, default 10
            Number of state probabilities p0..p_{n-1} to compute.
        on_unstable : {"raise", "flag"}, default "raise"
            ``"raise"`` stops at the first invalid or unstable row.
            ``"flag"`` records it with ``stable = False`` and missing results,
            so a sweep can safely cross the stability boundary.

        Returns
        -------
        pandas.DataFrame
            The input columns followed by a ``stable`` flag and the results.
            Also stored in ``result_df``.

        Raises
        ------
        MissingColumnsError
            If a required column is missing.
        InvalidParameterError
            If a row is invalid or unstable and ``on_unstable="raise"``.
        """
        if on_unstable not in ("raise", "flag"):
            raise ValueError("on_unstable must be 'raise' or 'flag'.")
        missing = [c for c in REQUIRED_COLUMNS[self._model] if c not in df.columns]
        if missing:
            raise MissingColumnsError(
                f"{self._model} needs columns {REQUIRED_COLUMNS[self._model]}; "
                f"missing {missing}."
            )

        rows = []
        for i, row in enumerate(df.itertuples(index=False)):
            params = {
                "lambd": row.lambd,
                "mu": row.mu,
                "s": int(row.s) if self._model == "mms" else 1,
                "cv": row.cv if self._model == "mg1" else 1.0,
            }
            try:
                rows.append({"stable": True, **compute_metrics(self._model, **params, n=n)})
            except InvalidParameterError as err:
                if on_unstable == "raise":
                    raise InvalidParameterError(f"Row {i}: {err}") from err
                rows.append({"stable": False})

        results = pd.DataFrame(rows)
        self._result_df = pd.concat([df.reset_index(drop=True), results], axis=1)
        return self._result_df

    @property
    def result_df(self) -> pd.DataFrame:
        """Results from the last call to ``experiment_df``."""
        if self._result_df is None:
            raise RuntimeError("Run experiment_df() before reading result_df.")
        return self._result_df


def find_parameter(model: str, vary: str, metric: str, target: float,
                   lower: float, upper: float, tol: float = 1e-10,
                   **fixed) -> float:
    """Find the parameter value at which a performance measure reaches a target.

    Uses bisection, so ``metric`` must cross ``target`` exactly once between
    ``lower`` and ``upper``. This gives an exact answer rather than reading
    the nearest value off a grid.

    Parameters
    ----------
    model : str
        ``"mm1"``, ``"mg1"`` or ``"mms"``.
    vary : str
        Name of the parameter to solve for, e.g. ``"mu"`` or ``"lambd"``.
    metric : str
        Performance measure to match, e.g. ``"Wq"``.
    target : float
        Required value of ``metric``.
    lower, upper : float
        Search interval for ``vary``; both ends must give a stable queue.
    tol : float, default 1e-10
        Width of the final interval.
    **fixed
        Other parameters of ``compute_metrics``, held constant.

    Returns
    -------
    float
        The value of ``vary`` at which ``metric == target``.

    Examples
    --------
    >>> round(find_parameter("mm1", "mu", "Wq", 0.5, 4.01, 10, lambd=4), 3)
    5.464
    """
    f: Callable[[float], float] = (
        lambda x: compute_metrics(model, **{**fixed, vary: x})[metric] - target
    )
    f_lo, f_hi = f(lower), f(upper)
    if np.sign(f_lo) == np.sign(f_hi):
        raise ValueError(f"{metric} does not cross {target} between {lower} and {upper}.")
    while upper - lower > tol:
        mid = (lower + upper) / 2
        if np.sign(f(mid)) == np.sign(f_lo):
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2
