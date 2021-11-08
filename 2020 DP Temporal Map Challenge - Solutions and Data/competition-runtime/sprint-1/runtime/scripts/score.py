import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from metric import Deid2Metric

INDEX_COLS = ["epsilon", "neighborhood", "year", "month"]

ROOT_DIRECTORY = Path("/codeexecution")
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_INCIDENTS = DATA_DIRECTORY / "incidents.csv"
DEFAULT_GROUND_TRUTH = DATA_DIRECTORY / "ground_truth.csv"
DEFAULT_SUBMISSION = ROOT_DIRECTORY / "submission.csv"


def get_ground_truth(incidents: pd.DataFrame, submission_format: pd.DataFrame):
    """ Aggregate the actual counts from the incidents.csv file """
    logger.debug("... creating pivot table")
    counts = incidents.assign(n=1).pivot_table(
        index=["neighborhood", "year", "month"],
        columns="incident_type",
        values="n",
        aggfunc=np.sum,
        fill_value=0,
    )
    # when you pivot, you only gets rows and columns for things that were actually there --
    # the ground truth may not have all of the neighborhoods, periods, or codes we expected to see,
    # so we'll fix that by reindexing and then filling the missing values
    epsilons = submission_format.index.levels[0]
    index_for_one_epsilon = submission_format.loc[epsilons[0]].index
    columns = submission_format.columns.astype(counts.columns)
    counts = (
        counts.reindex(columns=columns, index=index_for_one_epsilon)
        .fillna(0)
        .astype(np.int32)
    )
    logger.debug(
        "... duplicating the counts for every (neighborhood, year, month) to each epsilon"
    )
    ground_truth = submission_format.copy()
    for epsilon in epsilons:
        ground_truth.loc[epsilon] = counts.values
    return ground_truth


def main(
    incidents_csv: Path = DEFAULT_INCIDENTS,
    submission_csv: Path = DEFAULT_SUBMISSION,
    json_report: bool = False,
):
    """
    Given a raw incidents file and a submission file, calculate the score that the submission
    would receive if submitted.
    """
    for expected_file in (incidents_csv, submission_csv):
        if not expected_file.exists():
            raise typer.Exit(f"file not found: {expected_file}")

    logger.info(f"reading incidents from {incidents_csv} ...")
    incidents = pd.read_csv(incidents_csv, index_col=0)

    logger.info(f"reading submission from {submission_csv} ...")
    submission = pd.read_csv(submission_csv, index_col=INDEX_COLS)
    logger.info(f"read dataframe with {len(submission):,} rows")

    logger.info("computing ground truth ...")
    ground_truth = get_ground_truth(incidents, submission)
    logger.info(f"read dataframe with {len(ground_truth):,} rows")

    scorer = Deid2Metric()
    overall_score, row_scores = scorer.score(
        ground_truth.values, submission.values, return_individual_scores=True
    )
    logger.success(f"OVERALL SCORE: {overall_score}")

    if json_report:
        row_outcomes = []
        for idx, score in zip(submission.index, row_scores):
            epsilon, neighborhood, year, month = idx
            row_outcomes.append(
                {
                    "epsilon": epsilon,
                    "neighborhood": neighborhood,
                    "year": year,
                    "month": month,
                    "score": score,
                }
            )
        result = {"score": overall_score, "details": row_outcomes}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    typer.run(main)
