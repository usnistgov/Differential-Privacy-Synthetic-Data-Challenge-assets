import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"


def test_params():
    ground_truths = list(DATA_DIR.glob("sprint2*/**/ground_truth.csv"))
    parameters = {
        path: json.loads((path.parent / "parameters.json").read_text()) for path in ground_truths
    }
    kwargs = {
        path: json.loads((path.parent / "read_csv_kwargs.json").read_text())
        for path in ground_truths
    }
    for a, params_a in parameters.items():
        for b, params_b in parameters.items():
            if a == b:
                continue
            assert params_a["schema"]["YEAR"] == params_b["schema"]["YEAR"]

        df = pd.read_csv(a, **kwargs[a])
        ind_counts = df.sim_individual_id.value_counts()
        assert ind_counts.min() == 4
        ind_max = ind_counts.max()
        for run in params_a["runs"]:
            assert ind_max == run["max_records_per_individual"]
