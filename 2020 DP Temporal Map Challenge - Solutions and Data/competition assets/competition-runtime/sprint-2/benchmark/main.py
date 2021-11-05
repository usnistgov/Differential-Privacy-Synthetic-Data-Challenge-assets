import csv
import json
import random
from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import trange
import typer

ROOT_DIRECTORY = Path("/codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_GROUND_TRUTH = DATA_DIRECTORY / "ground_truth.csv"
DEFAULT_PARAMS = DATA_DIRECTORY / "parameters.json"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"


def simulate_row(parameters):
    """
    Naively create a valid row by picking random but valid values using the parameters file.
    """
    row = {}
    for col, d in parameters["schema"].items():
        value = 0
        if "values" in d:
            value = random.choice(d["values"])
        elif "min" in d:
            value = random.randint(d["min"], d["max"])
        elif col in {"INCTOT", "INCWAGE", "INCEARN"}:
            value = random.randint(0, 60_000)
        row[col] = value
    return row


def main(
    parameters_file: Path = DEFAULT_PARAMS,
    ground_truth_file: Path = DEFAULT_GROUND_TRUTH,
    output_file: Path = DEFAULT_OUTPUT,
    n_rows_to_simulate_per_epsilon: int = 20_000,
):
    """
    Create synthetic data appropriate to be submitted to the Sprint 2 competition.
    """
    logger.info(f"reading schema from {parameters_file} ...")
    with parameters_file.open("r") as fp:
        parameters = json.load(fp)

    ########################################################################################
    # NOTE: We don't actually look at the ground truth for this baseline other than to see #
    #       how many rows are present. You must ensure your solution is differentially     #
    #       private if you are using the ground truth.                                     #
    ########################################################################################
    logger.info(f"reading ground truth from {ground_truth_file} ...")
    dtypes = {
        column_name: d["dtype"] for column_name, d in parameters["schema"].items()
    }
    ground_truth = pd.read_csv(ground_truth_file, dtype=dtypes)
    logger.info(f"... read ground truth dataframe of shape {ground_truth.shape}")

    epsilons = [run["epsilon"] for run in parameters["runs"]]
    columns = list(parameters["schema"].keys())
    headers = ["epsilon"] + columns + ["sim_individual_id"]

    # start writing the CSV with headers
    logger.info(f"writing output to {output_file}")
    with output_file.open("w", newline="") as fp:
        output = csv.DictWriter(
            fp, fieldnames=headers, dialect="unix", quoting=csv.QUOTE_NONNUMERIC
        )
        output.writeheader()
        n_rows = 1
        for epsilon in epsilons:
            logger.info(f"starting simulation for epsilon={epsilon}")
            for i in trange(n_rows_to_simulate_per_epsilon):
                ################################################################################
                # NOTE: Naively assume only one row per individual (and lazily use iteration   #
                #       number as the simulated individual ID).                                #
                ################################################################################
                row = simulate_row(parameters)
                row["epsilon"] = epsilon
                row["sim_individual_id"] = i
                output.writerow(row)
                n_rows += 1
    logger.success(f"finished writing {n_rows:,} rows to {output_file}")

    logger.info("reading and writing one final time casting to correct dtypes ...")
    df = pd.read_csv(output_file)
    for col_name, d in parameters["schema"].items():
        df[col_name] = df[col_name].astype(d["dtype"])
    df.to_csv(output_file, index=False)
    logger.success(f"... done.")


if __name__ == "__main__":
    typer.run(main)
