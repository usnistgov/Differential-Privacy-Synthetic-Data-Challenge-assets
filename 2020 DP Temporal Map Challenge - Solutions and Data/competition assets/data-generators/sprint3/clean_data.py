import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from deid2.sprint3.metric import CATEGORICAL_COLS, NUMERIC_COLS, COL_TYPES

PROJECT_DIR = Path(__file__).parents[2]
MAX_RECORDS_PER_INDIVIDUAL = 200
DELTA = 1 / MAX_RECORDS_PER_INDIVIDUAL ** 2
MAX_RECORDS = 17_000_000

PAYMENT_TYPES = {
    "Unknown": -1,
    "Cash": 0,
    "Credit Card": 1,
    "Mobile": 2,
    "Prcard": 3,
    "No Charge": 5,
    "Dispute": 6,
    "Prepaid": 7,
    "Pcard": 8,
}

# ref: https://en.wikipedia.org/wiki/Community_areas_in_Chicago#List_of_community_areas
COMMUNITY_AREA_NAMES = {
    1: "Rogers Park",
    2: "West Ridge",
    3: "Uptown",
    4: "Lincoln Square",
    5: "North Center",
    6: "Lake View",
    7: "Lincoln Park",
    8: "Near North Side",
    9: "Edison Park",
    10: "Norwood Park",
    11: "Jefferson Park",
    12: "Forest Glen",
    13: "North Park",
    14: "Albany Park",
    15: "Portage Park",
    16: "Irving Park",
    17: "Dunning",
    18: "Montclare",
    19: "Belmont Cragin",
    20: "Hermosa",
    21: "Avondale",
    22: "Logan Square",
    23: "Humboldt Park",
    24: "West Town",
    25: "Austin",
    26: "West Garfield Park",
    27: "East Garfield Park",
    28: "Near West Side",
    29: "North Lawndale",
    30: "South Lawndale",
    31: "Lower West Side",
    32: "(The) Loop[11]",
    33: "Near South Side",
    34: "Armour Square",
    35: "Douglas",
    36: "Oakland",
    37: "Fuller Park",
    38: "Grand Boulevard",
    39: "Kenwood",
    40: "Washington Park",
    41: "Hyde Park",
    42: "Woodlawn",
    43: "South Shore",
    44: "Chatham",
    45: "Avalon Park",
    46: "South Chicago",
    47: "Burnside",
    48: "Calumet Heights",
    49: "Roseland",
    50: "Pullman",
    51: "South Deering",
    52: "East Side",
    53: "West Pullman",
    54: "Riverdale",
    55: "Hegewisch",
    56: "Garfield Ridge",
    57: "Archer Heights",
    58: "Brighton Park",
    59: "McKinley Park",
    60: "Bridgeport",
    61: "New City",
    62: "West Elsdon",
    63: "Gage Park",
    64: "Clearing",
    65: "West Lawn",
    66: "Chicago Lawn",
    67: "West Englewood",
    68: "Englewood",
    69: "Greater Grand Crossing",
    70: "Ashburn",
    71: "Auburn Gresham",
    72: "Beverly Hills",
    73: "Washington Heights",
    74: "Mount Greenwood",
    75: "Morgan Park",
    76: "O'Hare",
    77: "Edgewater",
    -1: "Unknown",
}

PARAMETERS = {
    "runs": [
        {
            "epsilon": 1.0,
            "delta": DELTA,
            "max_records": MAX_RECORDS,
            "max_records_per_individual": MAX_RECORDS_PER_INDIVIDUAL,
        },
        {
            "epsilon": 10.0,
            "delta": DELTA,
            "max_records": MAX_RECORDS,
            "max_records_per_individual": MAX_RECORDS_PER_INDIVIDUAL,
        },
    ],
    "schema": {},
}


def get_shift(hour_of_day):
    if hour_of_day >= 19 or hour_of_day < 3:
        return 0
    elif 11 <= hour_of_day < 19:
        return 1
    elif 3 <= hour_of_day < 11:
        return 2
    raise ValueError(f"bad hour of day: {hour_of_day}")


def main(
    input_file: Path, output_file: Path = None, random_seed: int = 42,
):
    logger.info("reading data ...")
    df = pd.read_csv(input_file, index_col=0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    logger.debug(f"... read in dataframe of shape {df.shape}")

    logger.info(f"converting `payment_type` to integer:\n{df.payment_type.value_counts()}")
    df["payment_type"] = df.payment_type.map(PAYMENT_TYPES).fillna(-1).astype(int)

    # narrow down to companies with enough data
    company_counts = df.company.value_counts()
    bad_companies = company_counts[company_counts < 100].index.values
    bad_company_mask = df.company.isin(bad_companies) | df.company.isnull()
    logger.info(f"dropping {bad_company_mask.sum():,} rows for taxi companies with N<100 or missing")
    df = df.loc[~bad_company_mask]
    logger.debug(f"... new shape: {df.shape}")

    # narrow down to taxis with enough data and not too much
    taxi_counts = df.taxi_week.value_counts()
    too_many = taxi_counts[taxi_counts > MAX_RECORDS_PER_INDIVIDUAL].index.values
    too_many_mask = df.taxi_week.isin(too_many)
    logger.info(
        f"dropping {too_many_mask.sum():,} taxi-weeks with more than {MAX_RECORDS_PER_INDIVIDUAL} rows"
    )
    df = df.loc[~too_many_mask]
    logger.debug(f"... new shape: {df.shape}")

    logger.info("assigning shift based on hour of day")
    for day in range(7):
        for hour in range(24):
            mask = (df.trip_hour_of_day == hour) & (df.trip_day_of_week == day)
            if day == 6 and hour >= 19:
                shift = 0
            else:
                shift = day * 3 + get_shift(hour)
            df.loc[mask, "shift"] = shift
    df["shift"] = df["shift"].astype(np.uint8)
    assert df["shift"].notnull().all()

    logger.info("creating a unique taxi ID from taxi_week")
    taxi_ids = {
        taxi_week: 2_000_000 + i for i, taxi_week in enumerate(sorted(df.taxi_week.unique()))
    }
    df["taxi_id"] = df.taxi_week.map(taxi_ids)

    logger.info("creating a unique company ID from company column")
    company_ids = {company: i for i, company in enumerate(sorted(df.company.unique()))}
    df["company_id"] = df.company.map(company_ids)

    # drop columns we don't need
    to_drop = ["tolls", "taxi_week", "company"]
    logger.info(f"dropping {to_drop} columns")
    df = df.drop(columns=to_drop)
    logger.debug(f"... new shape: {df.shape}")

    logger.info("rearranging columns, sorting data, and resetting index")
    sorted_cols = ["taxi_id"] + CATEGORICAL_COLS + NUMERIC_COLS
    df = df.loc[:, sorted_cols].sort_values(["taxi_id", "trip_day_of_week", "trip_hour_of_day"])

    for col in sorted_cols:
        for do_not_count in ["_id", "_seconds", "_miles", "_total", "fare", "tips"]:
            if do_not_count in col:
                continue
        n_col = df[col].nunique()
        values = ""
        if n_col <= 10:
            values = f" ({sorted(df[col].unique())})"
        logger.debug(f"unique values in column {col}: {n_col:,}{values}")

    for col in sorted_cols:
        minval = df[col].min()
        maxval = df[col].max()
        dtype = COL_TYPES.get(col, "int64")
        df[col] = df[col].astype(dtype)
        if col == "taxi_id":
            PARAMETERS["schema"][col] = {
                "dtype": "int64",
                "kind": "id",
                "min": 1_000_000,
                "max": 9_999_999,
            }
        elif col in CATEGORICAL_COLS:
            PARAMETERS["schema"][col] = {
                "dtype": df[col].dtype.name,
                "kind": "categorical",
                "values": [int(v) for v in sorted(df[col].unique())],
            }
        elif col in NUMERIC_COLS:
            PARAMETERS["schema"][col] = {
                "dtype": df[col].dtype.name,
                "kind": "numeric",
                "min": int(minval),
                "max": int(maxval),
            }
        else:
            logger.warning(f"not adding {col} to parameters.json")

    payment_type_names_by_value = {value: name for name, value in PAYMENT_TYPES.items()}
    values = PARAMETERS["schema"]["payment_type"]["values"]
    payment_type_names = [payment_type_names_by_value[value] for value in values]
    PARAMETERS["schema"]["payment_type"]["names"] = payment_type_names

    for col in ["pickup_community_area", "dropoff_community_area"]:
        values = PARAMETERS["schema"][col]["values"]
        PARAMETERS["schema"][col]["names"] = [COMMUNITY_AREA_NAMES[value] for value in values]

    # write out the data
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"writing {len(df):,} rows out to {output_file}")
        df.to_csv(output_file, index=False)

        parameters_file = output_file.parent / "parameters.json"
        logger.info(f"writing parameters.json out to {parameters_file}")
        parameters_file.write_text(json.dumps(PARAMETERS, indent=2))


if __name__ == "__main__":
    typer.run(main)
