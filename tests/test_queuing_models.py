"""Unit tests for the queuing_models package. Run with ``pytest``."""
import math

import pandas as pd
import pytest

from queuing_models import (
    InvalidModelError,
    InvalidParameterError,
    MissingColumnsError,
    Queue,
    compute_metrics,
    compute_p,
    find_parameter,
    validate_params,
)

MEASURES = ["Lq", "Ls", "Wq", "Ws", "p0"]


# --- Worked examples from the technical appendix ---------------------------

def test_mm1_worked_example():
    r = Queue("mm1").experiment(2, 3).results
    assert r["Ls"] == pytest.approx(2)
    assert r["Ws"] == pytest.approx(1)
    assert r["Wq"] == pytest.approx(2 / 3)


def test_mg1_worked_example():
    r = Queue("mg1").experiment(1.5, 2, cv=(20 / 60) / 0.5).results
    assert r["Lq"] == pytest.approx(1.625)
    assert r["Ls"] == pytest.approx(2.375)
    assert r["Wq"] == pytest.approx(1.625 / 1.5)
    assert r["Ws"] == pytest.approx(1.625 / 1.5 + 0.5)


def test_mm2_worked_example():
    r = Queue("mms").experiment(2, 3, s=2).results
    assert r["p0"] == pytest.approx(0.5)
    assert r["Lq"] == pytest.approx(1 / 12)
    assert r["Wq"] == pytest.approx(1 / 24)


# --- Internal consistency ---------------------------------------------------

@pytest.mark.parametrize("lam, mu, s", [(2, 3, 2), (9.5, 2, 5), (30, 1, 33), (45, 1, 50)])
def test_littles_formula(lam, mu, s):
    r = Queue("mms").experiment(lam, mu, s=s).results
    assert r["Ls"] == pytest.approx(lam * r["Ws"])
    assert r["Lq"] == pytest.approx(lam * r["Wq"])


@pytest.mark.parametrize("lam, mu", [(0.1, 1), (2, 3), (0.99, 1)])
def test_models_agree_in_limiting_cases(lam, mu):
    """M/G/1 with CV = 1 and M/M/s with s = 1 both reduce to M/M/1."""
    a = Queue("mm1").experiment(lam, mu).results
    b = Queue("mg1").experiment(lam, mu, cv=1).results
    c = Queue("mms").experiment(lam, mu, s=1).results
    for k in MEASURES:
        assert b[k] == pytest.approx(a[k])
        assert c[k] == pytest.approx(a[k])


def test_mm1_state_probabilities_match_formula():
    probs = compute_p(2, 3, 1, 10)
    assert probs == pytest.approx([(1 / 3) * (2 / 3) ** n for n in range(10)])


def test_mms_state_probabilities_sum_to_one():
    r = compute_metrics("mms", 9, 2, s=5, n=2000)
    assert sum(r[f"p{i}"] for i in range(2000)) == pytest.approx(1)


def test_large_number_of_servers_does_not_overflow():
    assert math.isfinite(Queue("mms").experiment(190, 1, s=200).results["Lq"])


def test_deterministic_service_halves_queue():
    mm1 = Queue("mm1").experiment(1.5, 2).results["Lq"]
    assert Queue("mg1").experiment(1.5, 2, cv=0).results["Lq"] == pytest.approx(mm1 / 2)


# --- Validation ---------------------------------------------------------------

@pytest.mark.parametrize("args", [
    (3, 3, 1, 1.0),            # rho = 1
    (10, 2, 5, 1.0),           # lambd = s * mu
    (4, 0, 1, 1.0),            # mu = 0
    (-1, 3, 1, 1.0),           # negative lambda
    (1, 2, 0, 1.0),            # zero servers
    (1, 2, 2.5, 1.0),          # non-integer servers
    (1, 2, 1, -3.0),           # negative cv
    (float("nan"), 3, 1, 1.0),
    ("2", 3, 1, 1.0),          # wrong type
])
def test_validate_params_rejects_bad_input(args):
    with pytest.raises(InvalidParameterError):
        validate_params(*args)


def test_custom_exceptions_are_value_errors():
    with pytest.raises(ValueError):
        Queue("mm1").experiment(3, 3)


def test_unknown_model_rejected():
    with pytest.raises(InvalidModelError):
        Queue("mm2")


def test_model_specific_parameters_enforced():
    with pytest.raises(InvalidParameterError):
        Queue("mm1").experiment(1, 2, s=2)      # M/M/1 has one server
    with pytest.raises(InvalidParameterError):
        Queue("mms").experiment(1, 2, s=2, cv=2)  # M/M/s assumes cv = 1


# --- Queue.experiment_df ------------------------------------------------------

def test_experiment_df_keeps_label_columns():
    df = pd.DataFrame({"operation": ["A", "B"], "lambd": [1, 2], "mu": [3, 3], "cv": [1, 2]})
    result = Queue("mg1").experiment_df(df)
    assert list(result["operation"]) == ["A", "B"]
    assert result["stable"].all()


def test_experiment_df_missing_columns():
    with pytest.raises(MissingColumnsError):
        Queue("mg1").experiment_df(pd.DataFrame({"lambd": [1], "mu": [2]}))


def test_experiment_df_raises_on_unstable_row_by_default():
    df = pd.DataFrame({"lambd": [2, 3], "mu": [3, 3]})
    with pytest.raises(InvalidParameterError, match="Row 1"):
        Queue("mm1").experiment_df(df)


def test_experiment_df_flags_unstable_rows():
    df = pd.DataFrame({"lambd": [2, 3, 4], "mu": [3, 3, 3]})
    result = Queue("mm1").experiment_df(df, on_unstable="flag")
    assert list(result["stable"]) == [True, False, False]
    assert result["Wq"].isna().sum() == 2


def test_result_df_before_experiment_raises():
    with pytest.raises(RuntimeError):
        Queue("mm1").result_df


# --- find_parameter -----------------------------------------------------------

def test_find_parameter_matches_closed_form_mm1():
    # Wq = lambda / (mu (mu - lambda)) = 0.5 with lambda = 4  ->  mu = 2 + sqrt(12)
    mu = find_parameter("mm1", "mu", "Wq", 0.5, lower=4.01, upper=10, lambd=4)
    assert mu == pytest.approx(2 + math.sqrt(12))


def test_find_parameter_matches_closed_form_mg1():
    # 2.5 * lambda / (22 * (22 - lambda)) = 4  ->  lambda = 1936 / 90.5
    lam = find_parameter("mg1", "lambd", "Wq", 4, lower=20, upper=21.99, mu=22, cv=2)
    assert lam == pytest.approx(1936 / 90.5)


def test_find_parameter_requires_a_crossing():
    with pytest.raises(ValueError):
        find_parameter("mm1", "mu", "Wq", 100, lower=5, upper=10, lambd=4)
