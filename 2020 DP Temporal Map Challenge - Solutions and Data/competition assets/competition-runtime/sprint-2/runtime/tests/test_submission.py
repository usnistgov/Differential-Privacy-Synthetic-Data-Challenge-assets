import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIRECTORY = Path("/codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
PARAMETERS_PATH = DATA_DIRECTORY / "parameters.json"
SUBMISSION_PATH = ROOT_DIRECTORY / "submission.csv"


@pytest.fixture(scope="session")
def submission():
    assert SUBMISSION_PATH.exists()
    return pd.read_csv(SUBMISSION_PATH)


@pytest.fixture(scope="session")
def parameters():
    with PARAMETERS_PATH.open("r") as fp:
        return json.load(fp)


def test_first_columns_is_epsilon(submission):
    assert (
        submission.columns[0] == "epsilon"
    ), "First column of submission must be 'epsilon'"


def test_last_column_is_sim_individual_id(submission):
    assert (
        submission.columns[-1] == "sim_individual_id"
    ), "Last column of submission must be 'sim_individual_id'"


def test_all_columns_match(submission, parameters):
    expected_data_columns = (
        ["epsilon"] + list(parameters["schema"].keys()) + ["sim_individual_id"]
    )
    found_data_columns = submission.columns.tolist()
    assert (
        found_data_columns == expected_data_columns
    ), f"Submission columns not as expected"


def test_submission_matches_schema(submission, parameters):
    for c, entry in parameters["schema"].items():
        assert (
            c in submission.columns
        ), f"expected column {i} to be {c} in data but it was not present"
        if "values" in entry:
            invalid_values = list(set(submission[c].tolist()) - set(entry["values"]))
            err_msg = f"column {c} contains invalid values '{invalid_values}' (accepted values '{entry['values']}')"
            assert not invalid_values, err_msg
        if "min" in entry:
            err_msg = f"column {c} contains values less than minimum ({entry['min']})"
            assert submission[c].min() >= entry["min"], err_msg
        if "max" in entry:
            err_msg = (
                f"column {c} contains values greater than maximum ({entry['max']})"
            )
            assert submission[c].max() <= entry["max"], err_msg


def test_max_rows_and_max_rows_per_individual(submission, parameters):
    # get parameters for the runs by epsilon value
    runs_df = pd.DataFrame(parameters["runs"]).set_index("epsilon")

    # calculate the sizes of each epsilon run and add to the df
    runs_df = pd.concat(
        [runs_df, submission.groupby("epsilon").size().rename("row_count")], axis=1
    )

    # max_records + delta are nan for epsilons in submission but not in parameters.json
    invalid_epsilons = runs_df[runs_df.max_records.isnull()].index.tolist()
    assert not invalid_epsilons, f"Submission has invalid epsilon: {invalid_epsilons}"

    # if the observed rows for an epsilon (row_count) larger than max records, error
    invalid_counts = runs_df[runs_df.row_count > runs_df.max_records].index.tolist()
    assert (
        not invalid_counts
    ), f"Some epsilon runs have too many individuals. Epsilons are: {invalid_counts}"


def test_epsilons_valid(submission, parameters):
    present_epsilons = set(submission["epsilon"].unique())
    expected_epsilons = set([run["epsilon"] for run in parameters["runs"]])

    missing_epsilons = expected_epsilons - present_epsilons
    assert not missing_epsilons, (
        f"Submission expected to have all epsilons {list(expected_epsilons)} but "
        f"the following are missing: {list(missing_epsilons)}"
    )

    extra_epsilons = present_epsilons - expected_epsilons
    assert (
        not extra_epsilons
    ), f"Submission has unexpected epsilons: {list(missing_epsilons)}"


def test_data_types_usable(submission, parameters):
    for c, entry in parameters["schema"].items():
        try:
            submission[c].astype(entry["dtype"])
        except (ValueError, TypeError):
            pytest.fail(f"Column {c} must be able to be cast to dtype {entry['dtype']}")


def test_all_float_values_are_finite(submission, parameters):
    for c, entry in parameters["schema"].items():
        dtype = entry["dtype"]
        if "float" in dtype:
            assert np.isfinite(
                submission[c].astype(dtype).values
            ).all(), f"Values in column {c} must be finite (not NaN or inf)"
