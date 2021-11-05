## DEID2 Sprint 2

### Setup

1. Set up a Python 3.9 virtual environment with requirements using
   `pip install -r requirements.txt`.
2. Ensure that `make` is installed.
3. Ensure the raw data from Knexus is present at `data/sprint2/raw/IL_OH_10Y_PUMS.csv`.

### Data pipeline

Run `make data`. This will take an hour or two, and requires a machine with 8 cores and at least 16 GB of memory.

The long-running part is the computationally intensive simulation of individuals. If you watch the logs, it will provide a URL to a status page (will look like `http://127.0.0.1:8787/status`) to monitor the status of the computation.

At the end, the final data will be available under `data/sprint2/final/public`.
