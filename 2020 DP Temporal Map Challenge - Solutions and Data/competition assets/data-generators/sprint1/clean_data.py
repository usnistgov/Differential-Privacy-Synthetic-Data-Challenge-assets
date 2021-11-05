import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

from loguru import logger
import numpy as np
import pandas as pd
import typer

from deid2.sprint1.settings import settings

PROJECT_DIR = Path(__file__).parents[2]


def clean_ground_truth_data(data_path: Path, codebook: dict) -> Tuple[pd.DataFrame, dict]:
    logger.info(f"reading raw data from {data_path.resolve()} ...")
    df = pd.read_csv(data_path, index_col=0)
    logger.info(f"read dataframe with {len(df):,} rows")

    periods = list(sorted([(d["year"], d["month"]) for d in codebook["schema"]["periods"]]))
    logger.info(f"narrowing down to wanted time periods: [{periods[0]}, {periods[-1]}] ...")
    mask = np.zeros(len(df), dtype=np.bool)
    for year, month in periods:
        mask |= (df.year == year) & (df.month == month)
    logger.info(f"dropping {len(df) - mask.sum():,} rows out of date range")
    df = df.loc[mask]

    neighborhood_names = [d["name"] for d in settings.get_parameters()["schema"]["neighborhood"]]
    mask = df.neighborhood.isin(neighborhood_names)
    if mask.any():
        unexpected = df.loc[~mask].neighborhood.unique()
        logger.info(
            f"dropping {len(df) - mask.sum():,} rows with unwanted neighborhoods: {unexpected}"
        )
        df = df.loc[mask]

    incident_type_names = [d["name"] for d in settings.get_parameters()["schema"]["incident_type"]]
    mask = df.incident_type.isin(incident_type_names)
    if mask.any():
        unexpected = df.loc[~mask].incident_type.unique()
        logger.info(
            f"dropping {len(df) - mask.sum():,} rows with unwanted incident types: {unexpected}"
        )
        df = df.loc[mask]

    logger.info("extract hour and minute from time")
    df["hour"] = df["time"].str[0:2].astype(int)
    df["minute"] = df["time"].str[3:5].astype(int)

    logger.info("narrowing down to desired columns")
    df = df[
        ["year", "month", "day", "hour", "minute", "neighborhood", "incident_type", "sim_resident"]
    ]

    return df.sort_values(["year", "month", "day", "hour", "minute"])


def encode_categoricals(incident_df: pd.DataFrame, codebook: dict) -> pd.DataFrame:
    neighborhood_codes = {d["name"]: d["code"] for d in codebook["schema"]["neighborhood"]}
    incident_codes = {d["name"]: d["code"] for d in codebook["schema"]["incident_type"]}
    return incident_df.assign(
        neighborhood=incident_df.neighborhood.map(neighborhood_codes),
        incident_type=incident_df.incident_type.map(incident_codes),
    )


def _get_default_output_path() -> Path:
    return settings.DATA_DIR / f"processed/clean_{datetime.utcnow().isoformat()}.csv"


def main(
    input_file: Path, output_file: Path = _get_default_output_path(),
):
    logger.debug(f"started with settings: {settings!r}")
    codebook = settings.get_parameters()

    logger.info("cleaning data")
    clean_data = clean_ground_truth_data(input_file, codebook=codebook)

    logger.info("encoding categorical variables")
    clean_data = encode_categoricals(clean_data, codebook=codebook)

    logger.info("creating codebook")
    n_individuals = clean_data.sim_resident.nunique()
    delta = 1.0 / (n_individuals ** 2)
    for run in codebook["runs"]:
        run["delta"] = delta
    codebook_file = output_file.parent / f"{output_file.stem}.json"
    logger.info(f"writing codebook out to {codebook_file}")
    with codebook_file.open("w") as fp:
        json.dump(codebook, fp, indent=2)

    # write out the data
    logger.info(f"writing {len(clean_data):,} rows out to {output_file}")
    clean_data.to_csv(output_file, index=True)


if __name__ == "__main__":
    typer.run(main)
