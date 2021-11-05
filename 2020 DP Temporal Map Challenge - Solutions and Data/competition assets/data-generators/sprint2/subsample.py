import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger


def main(
    ground_truth_file: Path, parameters_file: Path, output_path: Path, frac: float = 0.4, seed: int = 42,
):
    """
    Create synthetic data appropriate to be submitted to the Sprint 2 competition.
    """
    np.random.seed(seed)

    with parameters_file.open("r") as fp:
        parameters = json.load(fp)

    epsilons = [run["epsilon"] for run in parameters["runs"]]
    columns = list(parameters["schema"].keys())
    headers = ["epsilon"] + columns + ["sim_individual_id"]

    gt = pd.read_csv(ground_truth_file)
    individuals = gt.sim_individual_id.unique()
    n_take = int(len(individuals) * frac)
    logger.info(f"taking {n_take:,} of {len(individuals):,} individuals ...")

    n_rows = len(gt)
    individuals_take = np.random.choice(individuals, size=n_take)
    gt = gt.loc[gt.sim_individual_id.isin(individuals_take)]
    logger.info(f"filtered down to {len(gt):,} rows from {n_rows:,} before subsampling")

    result = pd.concat((gt.assign(epsilon=eps) for eps in epsilons), axis=0)
    logger.info(f"writing resulting df with epsilons {epsilons} [shape={result.shape}] to {output_path} ...")
    result.loc[:, headers].to_csv(output_path, index=False)
    logger.success("done")


if __name__ == "__main__":
    typer.run(main)
