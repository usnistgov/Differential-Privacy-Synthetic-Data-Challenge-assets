import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

import multiprocessing

COLS = {
    "PUMA": "str",
    "YEAR": "uint32",
    "HHWT": "float",
    "GQ": "uint8",
    "PERWT": "float",
    "SEX": "uint8",
    "AGE": "uint8",
    "MARST": "uint8",
    "RACE": "uint8",
    "HISPAN": "uint8",
    "CITIZEN": "uint8",
    "SPEAKENG": "uint8",
    "HCOVANY": "uint8",
    "HCOVPRIV": "uint8",
    "HINSEMP": "uint8",
    "HINSCAID": "uint8",
    "HINSCARE": "uint8",
    "EDUC": "uint8",
    "EMPSTAT": "uint8",
    "EMPSTATD": "uint8",
    "LABFORCE": "uint8",
    "WRKLSTWK": "uint8",
    "ABSENT": "uint8",
    "LOOKING": "uint8",
    "AVAILBLE": "uint8",
    "WRKRECAL": "uint8",
    "WORKEDYR": "uint8",
    "INCTOT": "int32",
    "INCWAGE": "int32",
    "INCWELFR": "int32",
    "INCINVST": "int32",
    "INCEARN": "int32",
    "POVERTY": "uint32",
    "DEPARTS": "uint32",
    "ARRIVES": "uint32",
}

BINS = {
    "AGE": np.r_[-np.inf, np.arange(20, 105, 5), np.inf],
    "INCTOT": np.r_[-np.inf, np.arange(0, 105_000, 5_000), np.inf],
    "INCWAGE": np.r_[-np.inf, np.arange(0, 105_000, 5_000), np.inf],
    "INCWELFR": np.r_[-np.inf, np.arange(0, 105_000, 5_000), np.inf],
    "INCINVST": np.r_[-np.inf, np.arange(0, 105_000, 5_000), np.inf],
    "INCEARN": np.r_[-np.inf, np.arange(0, 105_000, 5_000), np.inf],
    "POVERTY": np.r_[-np.inf, np.arange(0, 520, 20), np.inf],
    "HHWT": np.r_[-np.inf, np.arange(0, 520, 20), np.inf],
    "PERWT": np.r_[-np.inf, np.arange(0, 520, 20), np.inf],
    "DEPARTS": np.r_[
        -np.inf, [h * 100 + m for h in range(24) for m in (0, 15, 30, 45)], np.inf
    ],
    "ARRIVES": np.r_[
        -np.inf, [h * 100 + m for h in range(24) for m in (0, 15, 30, 45)], np.inf
    ],
}


def _apply_metric(counts):
    if counts.shape[1] < 2:
        return 2.0
    sums = counts.sum(axis=0)
    if np.min(sums) < 1:
        return 2.0
    return (counts / sums).diff(axis=1).sum(axis=1).abs().sum()


def _marginalize(grouped):
    counts = grouped.size().unstack("actual", fill_value=0)
    return counts.groupby(["PUMA", "YEAR"]).apply(_apply_metric).values.reshape(-1, 1)


class TidyFormatKMarginalMetric:
    """
    Implementation of k-marginal scoring
    """

    def __init__(
        self,
        raw_actual_df,
        raw_submitted_df,
        k,
        n_permutations,
        bins_for_numeric_cols=None,
        random_seed=None,
        verbose=False,
        bias_penalty_cutoff=250,
        processes=1,
    ):
        self.k = k
        self.n_permutations = n_permutations

        # combine the dataframes into one, groupable df
        self.combined_df = (
            pd.concat(
                [raw_actual_df.assign(actual=1), raw_submitted_df.assign(actual=0)]
            )
            .set_index(["PUMA", "YEAR"])
            .sort_index()
        )
        del raw_actual_df
        del raw_submitted_df

        # convert any numeric columns to bins
        self.bins_for_numeric_cols = bins_for_numeric_cols or {}
        for col, bins in bins_for_numeric_cols.items():
            self.combined_df[col] = pd.cut(self.combined_df[col], bins).cat.codes

        self.puma_year_index = self.combined_df.groupby(["PUMA", "YEAR"]).size().index
        self.n_puma_years = len(self.puma_year_index)

        self.bias_penalty_cutoff = bias_penalty_cutoff
        self.marginal_group_cols = list(
            sorted(set(COLS.keys()) - set(["PUMA", "YEAR"]))
        )
        self.random_seed = random_seed or 123456
        self.verbose = verbose
        self.processes = processes

    @staticmethod
    def _assert_sub_matches_schema(submission_df, parameters):
        schema = parameters["schema"]
        schema_errors = defaultdict(list)
        for c, conds in schema.items():
            if c not in submission_df.columns:
                schema_errors[c] += [
                    f"expected column {c} in data but it was not present"
                ]
                continue
            if "values" in conds:
                invalid_values = list(
                    set(submission_df[c].tolist()) - set(conds["values"])
                )
                if invalid_values:
                    schema_errors[c] += [
                        "invalid values '{}' (accepted values '{}')".format(
                            invalid_values, conds["values"],
                        )
                    ]

            if "min" in conds:
                if submission_df[c].min() < conds["min"]:
                    schema_errors[c] += [
                        "contains values less than minimum ({})".format(conds["min"],)
                    ]

            if "max" in conds:
                if submission_df[c].max() > conds["max"]:
                    schema_errors[c] += [
                        "contains values greater than maximum ({})".format(
                            conds["max"],
                        )
                    ]

        if schema_errors:
            errors = ";\n  ".join(
                "{} - {}".format(col, ", ".join(col_errs),)
                for col, col_errs in schema_errors.items()
            )
            raise ValueError("Errors were found in your submission:\n  " + errors)

    @staticmethod
    def _assert_sub_less_than_limit_and_epsilons_valid(submission_df, parameters):
        # get parameters for the runs by epsilon value
        runs_df = pd.DataFrame(parameters["runs"]).set_index("epsilon")

        # calculate the sizes of each epsilon run and add to the df
        runs_df = pd.concat(
            [runs_df, submission_df.groupby("epsilon").size().rename("row_count")],
            axis=1,
        )

        # max_records + delta are nan for epsilons in submission but not in parameters.json
        invalid_epsilons = runs_df[runs_df.max_records.isnull()].index.tolist()
        if invalid_epsilons:
            raise ValueError(
                "Submission has invalid epsilon: {}".format(invalid_epsilons)
            )

        # if the observed rows for an epsilon (row_count) larger than max records, error
        invalid_counts = runs_df[runs_df.row_count > runs_df.max_records].index.tolist()
        if invalid_counts:
            raise ValueError(
                "Some epsilon runs have too many individuals. Epsilons are: {}".format(
                    invalid_counts
                )
            )

        present_epsilons = set(submission_df["epsilon"].unique())
        expected_epsilons = set([run["epsilon"] for run in parameters["runs"]])
        missing_epsilons = expected_epsilons - present_epsilons
        if missing_epsilons:
            raise ValueError(
                f"Submission expected to have all epsilons {list(expected_epsilons)} "
                f"but the following are not present: {list(missing_epsilons)}"
            )

    def groupby_column_permutations(self):
        """
        Figure out which permutations of columns to use. Deterministic based on the random seed.
        """
        rand = np.random.RandomState(seed=self.random_seed)
        for _ in range(self.n_permutations):
            # grab the next set of K columns to marginalize
            features_i = rand.choice(
                self.marginal_group_cols, size=self.k, replace=False
            ).tolist()
            # we are going to group by the columns we always group by and also the K columns
            yield features_i

    def k_marginal_scores(self):
        # set up the columns we need to go through
        permutations_to_score = (
            self.combined_df.groupby(["PUMA", "YEAR"] + k_cols + ["actual"])
            for k_cols in self.groupby_column_permutations()
        )
        with multiprocessing.Pool(processes=self.processes) as pool:
            iters = pool.imap(_marginalize, permutations_to_score)
            if self.verbose:
                iters = tqdm(iters, total=self.n_permutations)
            scores = list(iters)
        return np.hstack(scores)

    def get_bias_mask(self):
        puma_year_counts = (
            self.combined_df.groupby(["PUMA", "YEAR", "actual"])
            .size()
            .unstack("actual", fill_value=0)
        )
        abs_errors = puma_year_counts.diff(axis=1).sum(axis=1).abs()
        self.bias_mask = abs_errors >= self.bias_penalty_cutoff
        return self.bias_mask

    def overall_score(self):
        # get the matrix of scores for each PUMA-YEAR (row) and k-col permutation (col)
        all_scores = self.k_marginal_scores()
        # take the row-wise mean to get the score per PUMA-YEAR
        self._scores = all_scores.mean(axis=1).ravel()
        # any individual PUMA-YEARs over the bias limit get the maximum penalty
        bias_mask = self.get_bias_mask()
        self._scores[bias_mask] = 2.0
        # return the mean of the scores per PUMA-YEAR for an overall score
        mean_score = self._scores.mean()
        nist_score = ((2.0 - mean_score) / 2.0) * 1_000
        return nist_score


def score_submission(
    ground_truth_csv: Path,
    submission_csv: Path,
    k: int = typer.Option(
        2, help="Number of columns (in addition to PUMA and YEAR) to marginalize on"
    ),
    n_permutations: int = typer.Option(
        50, help="Number of different permutations of columns to average"
    ),
    bias_penalty_cutoff: int = typer.Option(
        250,
        help="Absolute difference in PUMA-YEAR counts permitted before applying bias penalty",
    ),
    parameters_json: Path = typer.Option(
        None,
        help="Path to parameters.json; if provided, validates the submission using the schema",
    ),
    report_path: Path = typer.Option(
        None,
        help="Output path to save a JSON report file detailing scores at the PUMA-YEAR level",
    ),
    processes: int = typer.Option(None, help="Number of parallel processes to run"),
    verbose: bool = True,
):
    """
    Given the ground truth and a valid submission, compute the k-marginal score which the user would receive.
    """
    logger.info(f"reading in submission from {submission_csv}")

    try:
        submission_df = pd.read_csv(submission_csv, dtype=COLS)
    except TypeError as e:
        logger.error(f"Column {e} could not be read in as the expected data type")
        raise typer.Exit(1)

    if parameters_json is not None:
        logger.debug(f"validating submission ...")
        parameters = json.loads(parameters_json.read_text())
        logger.debug(f"checking that submission matches schema ...")
        TidyFormatKMarginalMetric._assert_sub_matches_schema(submission_df, parameters)
        logger.debug(
            f"checking that submission meets length limits and has proper epsilons ..."
        )
        TidyFormatKMarginalMetric._assert_sub_less_than_limit_and_epsilons_valid(
            submission_df, parameters
        )
        logger.success("... submission is valid ✓")

    logger.info(f"reading in ground truth from {ground_truth_csv}")
    ground_truth_df = pd.read_csv(ground_truth_csv, dtype=COLS)

    if "epsilon" not in submission_df.columns:
        submission_df["epsilon"] = None  # placeholder

    epsilons = submission_df["epsilon"].unique()
    scores_per_epsilon = []
    report = {"details": []}
    for epsilon in epsilons:
        epsilon_mask = submission_df.epsilon == epsilon
        n_rows = epsilon_mask.sum()
        logger.info(
            f"initializing metric for epsilon={epsilon} ({n_rows:,} rows of {len(submission_df):,} in submission)"
        )
        metric = TidyFormatKMarginalMetric(
            raw_actual_df=ground_truth_df,
            raw_submitted_df=submission_df.loc[epsilon_mask],
            k=k,
            n_permutations=n_permutations,
            bias_penalty_cutoff=bias_penalty_cutoff,
            bins_for_numeric_cols=BINS,
            verbose=verbose,
            processes=processes,
        )

        logger.info(f"starting calculation for epsilon={epsilon}")
        epsilon_score = metric.overall_score()
        if metric.bias_mask.sum():
            logger.warning(
                f"warning: {metric.bias_mask.sum()} PUMA-YEARs received a bias penalty"
            )

        # save out some records from this run if the user would like to output a report
        if report_path is not None:
            for i, ((puma, year), bias) in enumerate(metric.bias_mask.items()):
                nist_score = ((2.0 - metric._scores[i]) / 2.0) * 1_000
                record = {
                    "epsilon": epsilon,
                    "PUMA": puma,
                    "YEAR": year,
                    "score": nist_score,
                    "bias_penalty": bias,
                }
                report["details"].append(record)

        logger.success(f"score for epsilon {epsilon}: {epsilon_score}")
        scores_per_epsilon.append(epsilon_score)

    score_dict = dict(zip(epsilons, scores_per_epsilon))
    mean_score = np.mean(scores_per_epsilon)
    logger.success(
        f"finished scoring all epsilons: OVERALL SCORE = {mean_score} (per epsilon: {score_dict})"
    )

    if report_path is not None:
        with report_path.open("w") as fp:
            logger.info(f"writing out run report to {report_path}")
            json.dump(report, fp)
            logger.success(f"wrote out run report to {report_path}")
    return mean_score


if __name__ == "__main__":
    typer.run(score_submission)
