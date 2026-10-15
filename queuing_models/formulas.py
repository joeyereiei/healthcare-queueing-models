"""Steady-state formulas for the M/M/1, M/G/1 and M/M/s queuing models.

Rates share one time unit (e.g. per hour); times are returned in that unit.
``Ls``, ``Wq`` and ``Ws`` are derived from ``Lq`` with Little's formula.
"""
from .exceptions import InvalidModelError, InvalidParameterError
from .validation import validate_params

VALID_MODELS = ("mm1", "mg1", "mms")


def _check_model(model: str) -> str:
    """Return the lower-case model name, or raise if it is unknown."""
    m = str(model).lower()
    if m not in VALID_MODELS:
        raise InvalidModelError(f"model must be one of {VALID_MODELS} (got {model!r}).")
    return m


def _poisson_terms(tau: float, n_max: int) -> list[float]:
    """Return tau**n / n! for n = 0..n_max, built recursively to avoid overflow."""
    terms = [1.0]
    for n in range(1, n_max + 1):
        terms.append(terms[-1] * tau / n)
    return terms


def compute_rho(model: str, lambd: float, mu: float, s: int = 1) -> float:
    """Return the traffic intensity: lambd / mu, or lambd / (s * mu) for M/M/s."""
    return lambd / mu if _check_model(model) in ("mm1", "mg1") else lambd / (s * mu)


def compute_p0(model: str, lambd: float, mu: float, s: int = 1) -> float:
    """Return the probability that the system is empty."""
    rho = compute_rho(model, lambd, mu, s)
    if _check_model(model) in ("mm1", "mg1"):
        return 1 - rho
    terms = _poisson_terms(lambd / mu, s)
    return 1 / (sum(terms[:s]) + terms[s] / (1 - rho))


def compute_Lq(model: str, lambd: float, mu: float, s: int = 1, cv: float = 1.0) -> float:
    """Return the expected number of patients waiting in the queue.

    Parameters
    ----------
    model : str
        ``"mm1"``, ``"mg1"`` or ``"mms"``.
    lambd, mu : float
        Mean arrival rate and mean service rate per server.
    s : int, default 1
        Number of servers (M/M/s only).
    cv : float, default 1.0
        Coefficient of variation of service time (M/G/1 only).

    Returns
    -------
    float
        Lq from the model's steady-state equation.
    """
    model = _check_model(model)
    rho = compute_rho(model, lambd, mu, s)
    if model == "mm1":
        return rho ** 2 / (1 - rho)
    if model == "mg1":
        return (rho ** 2 / (1 - rho)) * ((1 + cv ** 2) / 2)
    p0 = compute_p0(model, lambd, mu, s)
    a_s_over_s_fact = _poisson_terms(lambd / mu, s)[s]          # (lambd/mu)^s / s!
    return p0 * a_s_over_s_fact * rho / (1 - rho) ** 2


def compute_Ls(lambd: float, mu: float, Lq: float) -> float:
    """Return the expected number in the system: Ls = Lq + lambd / mu."""
    return Lq + lambd / mu


def compute_Wq(lambd: float, Lq: float) -> float:
    """Return the expected waiting time in the queue: Wq = Lq / lambd."""
    return Lq / lambd


def compute_Ws(mu: float, Wq: float) -> float:
    """Return the expected time in the system: Ws = Wq + 1 / mu."""
    return Wq + 1 / mu


def compute_p(lambd: float, mu: float, s: int, n_max: int) -> list[float]:
    """Return [p0, p1, ..., p_{n_max - 1}] for an M/M/1 or M/M/s system.

    Parameters
    ----------
    lambd, mu : float
        Mean arrival rate and mean service rate per server.
    s : int
        Number of servers (1 for M/M/1).
    n_max : int
        Number of probabilities to return.

    Returns
    -------
    list of float
        p_n for n = 0..n_max - 1.
    """
    tau = lambd / mu
    p0 = compute_p0("mms", lambd, mu, s)
    terms = _poisson_terms(tau, min(n_max, s + 1))
    probs = []
    for n in range(n_max):
        if n <= s:
            probs.append(terms[n] * p0)
        else:
            probs.append(probs[-1] * tau / s)     # p_n = p_{n-1} * tau / s for n > s
    return probs


def compute_metrics(model: str, lambd: float, mu: float, s: int = 1,
                    cv: float = 1.0, n: int = 10) -> dict:
    """Validate the inputs and return every performance measure for one model run.

    Parameters
    ----------
    model : str
        ``"mm1"``, ``"mg1"`` or ``"mms"``.
    lambd, mu : float
        Mean arrival rate and mean service rate per server.
    s : int, default 1
        Number of servers. Must be 1 for M/M/1 and M/G/1.
    cv : float, default 1.0
        Coefficient of variation of service time. Must be 1 (exponential
        service) for M/M/1 and M/M/s.
    n : int, default 10
        Number of state probabilities p0..p_{n-1} to return (M/M/1, M/M/s).
        M/G/1 returns p0 only.

    Returns
    -------
    dict
        rho, Lq, Ls, Wq, Ws and p0 (plus p1.. for M/M/1 and M/M/s).

    Raises
    ------
    InvalidModelError
        If the model name is unknown.
    InvalidParameterError
        If the parameters are invalid, inconsistent with the model, or unstable.
    """
    model = _check_model(model)
    validate_params(lambd, mu, s, cv)
    if model in ("mm1", "mg1") and s != 1:
        raise InvalidParameterError(f"{model} has a single server; s must be 1 (got {s}).")
    if model in ("mm1", "mms") and cv != 1:
        raise InvalidParameterError(
            f"{model} assumes exponential service (cv = 1); use 'mg1' for cv = {cv}."
        )

    Lq = compute_Lq(model, lambd, mu, s, cv)
    Wq = compute_Wq(lambd, Lq)
    result = {
        "rho": compute_rho(model, lambd, mu, s),
        "Lq": Lq,
        "Ls": compute_Ls(lambd, mu, Lq),
        "Wq": Wq,
        "Ws": compute_Ws(mu, Wq),
    }
    if model == "mg1":
        result["p0"] = compute_p0(model, lambd, mu)
    else:
        result.update({f"p{i}": p for i, p in enumerate(compute_p(lambd, mu, s, max(n, 1)))})
    return result
