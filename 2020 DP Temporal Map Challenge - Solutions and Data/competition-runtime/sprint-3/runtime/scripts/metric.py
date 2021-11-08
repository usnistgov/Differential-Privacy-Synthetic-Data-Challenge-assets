import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm, trange

import multiprocessing

COL_TYPES = {
    "company_id": "int8",
    "dropoff_community_area": "int8",
    "fare": "int16",
    "payment_type": "int8",
    "pickup_community_area": "int8",
    "shift": "uint8",
    "tips": "int16",
    "trip_day_of_week": "int8",
    "trip_hour_of_day": "int8",
    "trip_miles": "int16",
    "trip_seconds": "int32",
    "trip_total": "int16",
}

CATEGORICAL_COLS = [
    "company_id",
    "pickup_community_area",
    "dropoff_community_area",
    "trip_day_of_week",
    "trip_hour_of_day",
    "shift",
    "payment_type",
]

NUMERIC_COLS = [
    "fare",
    "tips",
    "trip_total",
    "trip_seconds",
    "trip_miles",
]

BINS = {
    "fare": np.r_[-np.inf, np.arange(0, 100, step=10), np.inf],
    "tips": np.r_[-np.inf, np.arange(0, 100, step=10), np.inf],
    "trip_total": np.r_[-np.inf, np.arange(0, 100, step=10), np.inf],
    "trip_seconds": np.r_[-np.inf, np.arange(0, 2000, step=200), np.inf],
    "trip_miles": np.r_[-np.inf, np.arange(0, 100, step=10), np.inf],
}
ALWAYS_GROUP_BY = ["pickup_community_area", "shift"]
MARGINAL_COLS = [
    "company_id",
    "dropoff_community_area",
    "payment_type",
    "fare",
    "tips",
    "trip_total",
    "trip_seconds",
    "trip_miles",
]
PICKUP_DROPOFF_COLS = ["pickup_community_area", "dropoff_community_area"]

# kmarginal constants
K_MARGINAL_BIAS_PENALTY_CUTOFF = 250
PERMUTATIONS = [
    ALWAYS_GROUP_BY + [c1, c2]
    for c1 in MARGINAL_COLS
    for c2 in MARGINAL_COLS
    if c1 != c2
]
PRECALC_DIR = Path(tempfile.gettempdir()) / "kmarginal"
PRECALC_GT_DIR = PRECALC_DIR / "gt"
PRECALC_DP_DIR = PRECALC_DIR / "dp"

# higher order conjunction constants
HIGHER_ORDER_CONJUNCTION_ITERS = 50
MIN_HOC_DIFF = 5
MAX_HOC_DIFF = 50


def bin_numerics(df):
    for col in df.columns:
        if col in BINS:
            df.loc[:, col] = pd.cut(df[col], BINS[col], right=False, labels=False)
            if len(BINS[col]) < 255:
                df.loc[:, col] = df[col].astype(np.uint8)
    return df


def _get_counts(perm):
    filename = "-".join(perm)
    index_cols = list(range(len(perm)))
    dp = pd.read_csv(PRECALC_DP_DIR / filename, index_col=index_cols)
    gt = pd.read_csv(PRECALC_GT_DIR / filename, index_col=index_cols)
    counts = dp.join(gt, how="outer").fillna(0).astype(np.int)
    return counts


def _kmarginal_from_precomputed(perm):
    counts = _get_counts(perm)
    return counts.groupby(ALWAYS_GROUP_BY).apply(_apply_metric).rename("-".join(perm))


def _apply_metric(counts):
    if counts.shape[1] < 2:
        return 2.0
    sums = counts.sum(axis=0)
    if np.min(sums) < 1:
        return 2.0
    return (counts / sums).diff(axis=1).sum(axis=1).abs().sum()


class TidyFormatKMarginalMetric:
    """
    Implementation of k-marginal scoring
    """

    def __init__(
        self, raw_actual_df, raw_submitted_df, random_seed=None, processes=1,
    ):
        self.random_seed = random_seed or 123456
        self.processes = processes
        self.report = {}

        # combine the dataframes into one, groupable df
        self.ground_truth = raw_actual_df
        self.submitted = raw_submitted_df
        PRECALC_GT_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"created working directory at {PRECALC_DIR}")
        if PRECALC_DP_DIR.exists():
            logger.warning("found existing submission counts; removing")
            shutil.rmtree(PRECALC_DP_DIR)
        PRECALC_DP_DIR.mkdir()

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

    def _precompute_marginal_counts(self):
        todo = PERMUTATIONS + [
            # for the pickup/dropoff part of the metric
            PICKUP_DROPOFF_COLS,
            # for bias mask
            ALWAYS_GROUP_BY,
        ]
        logger.info("precomputing ground truth counts for each permutations ...")
        for perm in tqdm(todo):
            filename = "-".join(perm)
            gt_path = PRECALC_GT_DIR / filename
            if gt_path.exists():
                # avoid doing this if it has already been done
                continue
            self.ground_truth.loc[:, perm].value_counts().rename(1).to_csv(gt_path)
        logger.info("precomputing submitted counts for each permutation ...")
        for perm in tqdm(todo):
            filename = "-".join(perm)
            dp_path = PRECALC_DP_DIR / filename
            self.submitted.loc[:, perm].value_counts().rename(0).to_csv(dp_path)

    def k_marginal_scores(self):
        self._precompute_marginal_counts()
        logger.info(
            f"running k-marginal count comparisons in parallel with {self.processes} processes..."
        )
        with multiprocessing.Pool(processes=self.processes) as pool:
            iters = pool.imap(_kmarginal_from_precomputed, PERMUTATIONS)
            iters = tqdm(iters, total=len(PERMUTATIONS))
            scores = list(iters)
        # join all the place/time-respective scores horizontally into a dataframe with a row for
        # each place/time and a column for each of the k-marginal permutations
        score_df = pd.concat(scores, axis=1, ignore_index=False, join="outer")
        return score_df

    def scaled_k_marginal_score(self):
        # get the matrix of scores for each place/time (row) and k-col permutation (col)
        score_df = self.k_marginal_scores()
        # take the row-wise mean to get the score per place/time
        self._scores = score_df.mean(axis=1)
        # any individual place/time over the bias limit get the maximum penalty
        bias_mask = self.get_bias_mask()
        bias_idxs = bias_mask[bias_mask].index
        if bias_mask.sum():
            logger.warning(
                f"warning: {bias_mask.sum()} place/times received a bias penalty"
            )
        self._scores.loc[bias_idxs] = 2.0
        # get the mean of the scores per place/time for an overall score
        raw_score = self._scores.mean()
        # scale to [0, 1] and reverse direction so higher is better
        scaled_score = (2.0 - raw_score) / 2.0
        return scaled_score

    def get_bias_mask(self):
        counts = _get_counts(ALWAYS_GROUP_BY)
        abs_errors = counts.diff(axis=1).sum(axis=1).abs()
        self.bias_mask = abs_errors >= K_MARGINAL_BIAS_PENALTY_CUTOFF
        return self.bias_mask

    def pickup_dropoff_score(self):
        counts = _get_counts(PICKUP_DROPOFF_COLS)
        raw_score = _apply_metric(counts)
        # scale to [0, 1] and reverse direction so higher is better
        scaled_score = (2.0 - raw_score) / 2.0
        return scaled_score

    def higher_order_conjunction(self, n_iters=HIGHER_ORDER_CONJUNCTION_ITERS):
        """
        Competition-specific implementation of the Higher Order Conjunction metric.
        """

        def _count_shift_and_pickup_areas(df):
            """
            For each individual (row), count up the number of times each shift is observed
            and each pickup location is observed and concatenate them into a single count
            dataframe representing the "kind" of individual this is by WHEN they work (shift) and
            WHERE they work (pickup_community_area).
            """
            by_shift = pd.pivot_table(
                df.assign(n=1),
                values="n",
                index="taxi_id",
                columns="shift",
                aggfunc="count",
                fill_value=0,
            )
            by_pickup = pd.pivot_table(
                df.assign(n=1),
                values="n",
                index="taxi_id",
                columns="pickup_community_area",
                aggfunc="count",
                fill_value=0,
            )
            return by_shift.join(by_pickup, rsuffix="p")

        def _count_up_how_many_rows_are_similar(raw_counts, arr, max_diffs):
            """
            Given a pivot matrix of counts and a single row, figure out how many rows are within the allowable
            limits of similarity for each column.
            """
            # let M be the number of taxi IDs and N be the number of columns
            abs_err = np.abs(raw_counts - arr)  # MxN matrix of absolute diffs from arr
            cells_within_limit = (
                abs_err <= max_diffs
            )  # MxN matrix of booleans saying whether diff is allowable
            entire_row_within_limits = cells_within_limit.all(
                axis=1
            )  # Mx1 array of bools for whether row is similar
            n_rows_within_limits = (
                entire_row_within_limits.sum()
            )  # scalar count of how many rows were similar
            return n_rows_within_limits

        # set a random seed for reproducibility
        rng = np.random.RandomState(seed=self.random_seed)

        # pivot the counts for the ground truth
        pivoted_gt = _count_shift_and_pickup_areas(self.ground_truth)
        n_gt, n_cols = pivoted_gt.shape
        assert n_cols == 99
        # pivot the counts for the privatized
        pivoted_dp = _count_shift_and_pickup_areas(self.submitted)
        # make sure the submitted has the same columns as the ground truth
        pivoted_dp = (
            pivoted_dp.reindex(columns=pivoted_gt.columns)
            .fillna(0)
            .values.astype(np.int)
        )
        pivoted_gt = pivoted_gt.values
        n_dp, n_cols = pivoted_dp.shape
        assert n_cols == 99

        # initialize holders for counting how many individuals are similar in each data set
        n_similar_gt = np.zeros(n_iters, dtype=np.int)
        n_similar_dp = np.zeros(n_iters, dtype=np.int)
        for i in trange(n_iters):
            # choose an individual from the ground truth to represent an archetypal individual
            random_individual = pivoted_gt[rng.randint(low=0, high=n_gt)]
            # come up with varying "difficulties" for each feature of the count vector to count as similar
            # to the randomly selected individual
            max_diff_to_qualify_as_similar = rng.randint(
                MIN_HOC_DIFF, MAX_HOC_DIFF + 1, size=n_cols
            )
            # number of rows in the ground truth "like" this row
            n_similar_gt[i] = _count_up_how_many_rows_are_similar(
                pivoted_gt, random_individual, max_diff_to_qualify_as_similar
            )
            # number of rows in the privatized "like" this row
            n_similar_dp[i] = _count_up_how_many_rows_are_similar(
                pivoted_dp, random_individual, max_diff_to_qualify_as_similar
            )

        # normalize the absolute counts into proportions of all individuals who are similar
        prop_gt = n_similar_gt / n_gt
        prop_dp = n_similar_dp / n_dp
        logger.debug(
            f"proportion errors for each of {n_iters} iterations: {(prop_gt - prop_dp).round(4)}"
        )
        # calculate the MAE in [0, 1] where lower is better
        mean_absolute_error = np.abs(prop_gt - prop_dp).mean()
        # reverse direction to higher is better
        return 1.0 - mean_absolute_error

    def overall_score(self):
        logger.info("computing k-marginals...")
        k_marginal_score = self.scaled_k_marginal_score()
        self.report["k_marginal_score"] = k_marginal_score
        logger.success(f"RESULT [KMARGINAL]: {k_marginal_score}")

        logger.info("computing pickup-dropoff marginal...")
        pickup_dropoff_score = self.pickup_dropoff_score()
        self.report["pickup_dropoff_score"] = pickup_dropoff_score
        logger.success(f"RESULT [SPATIAL]: {pickup_dropoff_score}")

        logger.info("computing higher order conjunction...")
        higher_order_conjunction_score = self.higher_order_conjunction()
        self.report["higher_order_conjunction"] = higher_order_conjunction_score
        logger.success(f"RESULT [HOC]: {higher_order_conjunction_score}")

        overall_score = 1000.0 * np.mean(
            [k_marginal_score, pickup_dropoff_score, higher_order_conjunction_score]
        )
        return overall_score


def score_submission(
    ground_truth_csv: Path,
    submission_csv: Path,
    parameters_json: Path = typer.Option(
        None,
        help="Path to parameters.json; if provided, validates the submission using the schema",
    ),
    report_path: Path = typer.Option(
        None,
        help="Output path to save a JSON report file detailing scores at the place/time level",
    ),
    processes: int = typer.Option(None, help="Number of parallel processes to run"),
):
    """
    Given the ground truth and a valid submission, compute the k-marginal score which the user would receive.
    """
    logger.info(f"reading in submission from {submission_csv}")

    try:
        submission_df = pd.read_csv(submission_csv, dtype=COL_TYPES)
    except TypeError as e:
        logger.error(f"Column {e} could not be read in as the expected data type")
        raise typer.Exit(1)

    if parameters_json is not None:
        logger.debug("validating submission ...")
        parameters = json.loads(parameters_json.read_text())
        logger.debug("checking that submission matches schema ...")
        TidyFormatKMarginalMetric._assert_sub_matches_schema(submission_df, parameters)
        logger.debug(
            "checking that submission meets length limits and has proper epsilons ..."
        )
        TidyFormatKMarginalMetric._assert_sub_less_than_limit_and_epsilons_valid(
            submission_df, parameters
        )
        logger.success("... submission is valid ✓")

    logger.debug("binning submission")
    submission_df = bin_numerics(submission_df)

    logger.info(f"reading in ground truth from {ground_truth_csv}")
    ground_truth_df = pd.read_csv(ground_truth_csv, dtype=COL_TYPES)
    logger.debug("binning ground truth")
    ground_truth_df = bin_numerics(ground_truth_df)

    if "epsilon" not in submission_df.columns:
        submission_df["epsilon"] = None  # placeholder

    epsilons = submission_df["epsilon"].unique()
    scores_per_epsilon = []
    report = {"details": [], "per_epsilon": []}
    for epsilon in epsilons:
        epsilon_mask = submission_df.epsilon == epsilon
        n_rows = epsilon_mask.sum()
        logger.info(
            f"initializing metric for epsilon={epsilon} ({n_rows:,} rows of {len(submission_df):,} in submission)"
        )
        metric = TidyFormatKMarginalMetric(
            raw_actual_df=ground_truth_df,
            raw_submitted_df=submission_df.loc[epsilon_mask, :],
            processes=processes,
        )

        logger.info(f"starting calculation for epsilon={epsilon}")
        epsilon_score = metric.overall_score()

        # save out some records from this run if the user would like to output a report
        if report_path is not None:
            report["per_epsilon"].append(metric.report)
            for (place, time), bias in metric.bias_mask.items():
                nist_score = ((2.0 - metric._scores.loc[(place, time)]) / 2.0) * 1_000
                record = {
                    "epsilon": epsilon,
                    "place": place,
                    "time": time,
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
