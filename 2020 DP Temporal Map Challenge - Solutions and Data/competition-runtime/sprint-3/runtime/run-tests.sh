#!/bin/bash
set -e

echo "Running Python tests"
conda run -n py-$1 python tests/test_installs.py

echo "Running R tests"
conda run -n r-$1 Rscript tests/test_installs.R
