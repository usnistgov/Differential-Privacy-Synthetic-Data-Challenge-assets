from pathlib import Path

import dask.distributed
import dask.dataframe as dd
from loguru import logger
import pandas as pd
import typer

PROJECT_DIR = Path(__file__).parents[2]

FIXED_TRAITS = ["SEX", "RACE", "HISPAN"]
MARRIAGE_TRANSITIONS = {
    1: (1, 2, 3, 4, 5),
    2: (1, 2,),
    3: (3, 4,),
    4: (4,),
    5: (5,),
    6: (1, 2, 6),
}

CITIZEN_TRANSITIONS = {
    0: (0,),
    1: (1,),
    2: (2,),
    3: (2, 3, 4),
    4: (2, 4),
    5: (2, 3, 4, 5,),
}


def simulate_individuals_groupby(subdf, max_year=2018):
    """
    Take a DataFrame grouped by the non-negotiable features and then loop through it.
    For each row, start at that row's year and iterate through the remaining years in
    the data to narrow down all the other rows which haven't already been assigned
    and see if they would work as subsequent data for the same individual.
    """
    # initialize the mapping
    subdf["sim_individual_id"] = -1

    # for each row in the grouped dataframe
    for row_id in subdf.sort_values("YEAR").index.values:
        if subdf.loc[row_id, "sim_individual_id"] > -1:
            # if already assigned then skip the row - we did it already
            continue

        # initialize the state variables
        individual = row_id
        subdf.loc[row_id, "sim_individual_id"] = row_id
        row = subdf.loc[row_id]

        # for all the years left, see if there are more rows that could be this person
        for year in range(int(row.YEAR) + 1, max_year + 1):
            # filter down to appropriate rows
            delta = year - row.YEAR
            # filter down to the year in question
            mask = subdf.YEAR == year
            # only people who have not already been assigned
            mask &= subdf.sim_individual_id < 0
            # filter down to the right age
            mask &= subdf.AGE == (int(row.AGE) + delta)
            # education monotonically increasing
            mask &= subdf.EDUC >= int(row.EDUC)
            # education doesn't skip too many levels
            mask &= subdf.EDUC <= int(row.EDUC) + delta + 1
            # marriage status transition makes sense
            mask &= subdf.MARST.isin(MARRIAGE_TRANSITIONS.get(int(row.MARST), (1, 2, 3, 4, 5, 6)))
            # speaks english makes sense
            mask &= subdf.SPEAKENG >= row.SPEAKENG
            # citizenship status makes sense
            mask &= subdf.CITIZEN.isin(
                CITIZEN_TRANSITIONS.get(int(row.CITIZEN), (1, 2, 3, 4, 5, 6))
            )
            candidates = subdf.loc[mask]

            # if any candidates are in the same PUMA, narrow down to just those irrespective
            # of candidates with closer incomes outside the PUMA
            puma_mask = candidates.PUMA == row.PUMA
            if puma_mask.any():
                candidates = candidates.loc[puma_mask]

            # ... now choose the row with the most similar income
            if len(candidates):
                new_row_id = (candidates.INCTOT - row.INCTOT).abs().idxmin()
                # tag this as belonging to the same individual
                subdf.loc[new_row_id, "sim_individual_id"] = individual
                # add to our mapping
                row_id, row = new_row_id, subdf.loc[new_row_id]

    return subdf


def main(
    input_file: Path,
    output_file: Path,
    cloud: bool = False,
    frac: float = None,
    n_workers: int = 8,
):
    logger.info("creating dask cluster ...")
    if cloud:
        import coiled

        cluster = coiled.Cluster(n_workers=n_workers, configuration="coiled/default-py38")
    else:
        cluster = dask.distributed.LocalCluster(n_workers=n_workers)

    logger.info("creating dask client ...")
    client = dask.distributed.Client(cluster)
    logger.info(f"dask dashboard {client.dashboard_link}")

    logger.info("reading in data ...")
    df = pd.read_csv(input_file, index_col=0)
    if frac is not None:
        logger.info(f"... taking a subsample of {frac:%} of rows")
        df = df.sample(frac=frac)
    ddf = dd.from_pandas(df, npartitions=100)
    ddf.persist()

    logger.info("running simulation ...")
    final_df = ddf.groupby(FIXED_TRAITS).apply(simulate_individuals_groupby).compute()

    logger.success(f"writing output to {output_file} ...")
    final_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    typer.run(main)
