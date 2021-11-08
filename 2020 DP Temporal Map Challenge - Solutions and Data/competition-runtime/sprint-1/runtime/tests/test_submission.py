import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIRECTORY = Path("/codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
PARAMETERS_PATH = DATA_DIRECTORY / "parameters.json"
INCIDENT_DF_PATH = DATA_DIRECTORY / "incidents.csv"
SUBMISSION_PATH = ROOT_DIRECTORY / "submission.csv"


@pytest.fixture(scope="session")
def submission():
    assert SUBMISSION_PATH.exists()
    return pd.read_csv(
        SUBMISSION_PATH, index_col=["epsilon", "neighborhood", "year", "month"]
    )


@pytest.fixture(scope="session")
def parameters():
    with PARAMETERS_PATH.open("r") as fp:
        return json.load(fp)


def test_shape_same(submission, parameters):
    n_neighborhoods = len(parameters["schema"]["neighborhood"])
    n_periods = len(parameters["schema"]["periods"])
    n_epsilons = len(parameters["runs"])

    n_rows = n_epsilons * n_neighborhoods * n_periods
    n_cols = len(parameters["schema"]["incident_type"])
    assert submission.shape == (n_rows, n_cols)


def test_index_matches(submission, parameters):
    neighborhood_codes = [d["code"] for d in parameters["schema"]["neighborhood"]]
    periods = [(d["year"], d["month"]) for d in parameters["schema"]["periods"]]
    epsilons = [d["epsilon"] for d in parameters["runs"]]

    i = 0
    for epsilon in epsilons:
        for neighborhood in neighborhood_codes:
            for year, month in periods:
                print(year, month)
                actual = submission.index.values[i]
                assert actual == (epsilon, neighborhood, year, month)
                i += 1


def test_columns_match(submission, parameters):
    expected_cols = [str(d["code"]) for d in parameters["schema"]["incident_type"]]
    actual_cols = submission.columns.astype(str).values.tolist()
    assert actual_cols == expected_cols


def test_data_types_match(submission):
    acceptable_dtypes = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
    for column in submission.columns:
        assert submission[column].dtype in acceptable_dtypes


def test_all_values_are_finite(submission):
    assert np.isfinite(
        submission.values
    ).all(), "Count values must be finite (not NaN or inf)"


def test_all_values_are_nonzero(submission):
    assert (submission.values >= 0).all(), "Count values must be non-negative"
