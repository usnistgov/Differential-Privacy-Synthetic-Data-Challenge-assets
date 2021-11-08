import json
import warnings
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer


# current numpy and pandas versions have a FutureWarning for a mask operation we use;
# we will ignore it
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIRECTORY = Path("/codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_SUBMISSION_FORMAT = DATA_DIRECTORY / "submission_format.csv"
DEFAULT_INCIDENTS = DATA_DIRECTORY / "incidents.csv"
DEFAULT_PARAMS = DATA_DIRECTORY / "parameters.json"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"


def naively_add_laplace_noise(arr, scale: float, seed: int = None):
    """
    Add Laplace random noise of the desired scale to the dataframe of counts. Noise will be
    clipped to [0,∞) and rounded to the nearest positive integer.

    Expects a numpy array and a Laplace scale.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.laplace(scale=scale, size=arr.size).reshape(arr.shape)
    result = np.clip(arr + noise, a_min=0, a_max=np.inf)
    return result.round().astype(np.int)


def get_ground_truth(incidents: pd.DataFrame, submission_format: pd.DataFrame):
    # get actual counts
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
    submission_format: Path = DEFAULT_SUBMISSION_FORMAT,
    incident_csv: Path = DEFAULT_INCIDENTS,
    output_file: Optional[Path] = DEFAULT_OUTPUT,
    params_file: Path = DEFAULT_PARAMS,
):
    """
    Generate an example submission. The defaults are set so that the script will run successfully
    without being passed any arguments, invoked only as `python main.py`.
    """
    logger.info("loading parameters")
    params = json.loads(params_file.read_text())
    # calculate the Laplace scales for each run
    scales = {
        run["epsilon"]: run["max_records_per_individual"] / run["epsilon"]
        for run in params["runs"]
    }
    logger.info(f"laplace scales for each epsilon: {scales}")

    # read in the submission format
    logger.info(f"reading submission format from {submission_format} ...")
    submission_format = pd.read_csv(
        submission_format, index_col=["epsilon", "neighborhood", "year", "month"]
    )
    logger.info(f"read dataframe with {len(submission_format):,} rows")

    # read in the raw incident data
    logger.info(f"reading raw incident data from {incident_csv} ...")
    incidents = pd.read_csv(incident_csv, index_col=0)
    logger.info(f"read dataframe with {len(incidents):,} rows")

    logger.info("counting up incidents by (neighborhood, year, month)")
    # get actual counts (BUT DO NOT "LOOK" AT THEM YET -- see ALL CAPS note below)
    ground_truth = get_ground_truth(incidents, submission_format)

    logger.info(f"privatizing each set of {len(submission_format):,} counts...")
    submission = submission_format.copy()
    with tqdm(total=len(submission_format)) as pbar:
        # for each of the neighboorhood ...
        for (epsilon, neighborhood, year, month) in submission_format.index.values:
            ###############################################################################
            # NOTE: THIS IS THE DIFFERENTIAL-PRIVACY SENSITIVE PORTION OF THIS OPERATION  #
            #       WHERE WE ARE "LOOKING" AT REAL VALUES. UP UNTIL THIS OPERATION, WE'VE #
            #       ONLY USED THE KNOWN PARAMETERS OF THE PROBLEM. _YOU_ MUST ENSURE THAT #
            #       YOUR IMPLEMENTATION MATCHES THE PROOF THAT YOU SUBMIT, DOES NOT VIO-  #
            #       LATE THE ASSUMPTIONS OF DIFFERENTIAL PRIVACY, AND COMPENSATES FOR ANY #
            #       ADDITIONAL "PEEKING" AT THE NON-PRIVATIZED DATA                       #
            ###############################################################################
            actual_counts = ground_truth.loc[
                (epsilon, neighborhood, year, month)
            ].values

            # we will add some naive Laplace noise as an example of the most
            # basic privatization operation we can do
            privatized_counts = naively_add_laplace_noise(
                actual_counts, scale=scales[epsilon]
            )

            # put these counts in the submission dataframe
            submission.loc[(epsilon, neighborhood, year, month)] = privatized_counts

            # update the progress bar
            pbar.update(1)

    if output_file is not None:
        logger.info(f"writing {len(submission_format):,} rows out to {output_file}")
        submission.to_csv(output_file, index=True)

    return submission_format


if __name__ == "__main__":
    typer.run(main)
