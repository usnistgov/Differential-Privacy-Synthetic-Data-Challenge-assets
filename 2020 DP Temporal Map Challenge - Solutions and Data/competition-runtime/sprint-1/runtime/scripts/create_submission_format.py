import json
from pathlib import Path
import sys

from loguru import logger
import pandas as pd
import typer


def get_submission_format(params: dict):
    """
    Create a dataframe with a MultiIndex for {neighborhood, year, month, incident} keys and
    one column as the value.
    """
    neighborhood_codes = [d["code"] for d in params["schema"]["neighborhood"]]
    incident_codes = [d["code"] for d in params["schema"]["incident_type"]]
    periods = [(d["year"], d["month"]) for d in params["schema"]["periods"]]
    epsilons = [d["epsilon"] for d in params["runs"]]

    indices = []
    for epsilon in epsilons:
        for neighborhood in neighborhood_codes:
            for year, month in periods:
                indices.append((epsilon, neighborhood, year, month))
    index = pd.MultiIndex.from_tuples(
        indices, names=["epsilon", "neighborhood", "year", "month"]
    )
    df = pd.DataFrame(index=index, columns=incident_codes).fillna(0)
    return df


def main(params_file: Path, output_file: Path = None):
    """
    Create a barebones submission file with all zeros for counts.

    If `--output-file` is not provided then the script will print output to stdout.
    """
    if not params_file.exists():
        raise typer.Exit(f"file not found: {params_file}")

    logger.info("loading parameters")
    params = json.loads(params_file.read_text())

    logger.info("creating submission format")
    submission_format = get_submission_format(params)

    output_file = output_file or sys.stdout
    logger.info(f"writing {len(submission_format):,} rows out to {output_file}")
    submission_format.to_csv(output_file, index=True)


if __name__ == "__main__":
    typer.run(main)
