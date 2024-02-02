#!/bin/bash

set -e

# preprocess.sh: this script should also be included for reproducibility.
# It setups computation that usually run only one time, such as creating a document database.

ROOT=$(dirname "$0")
cd "$ROOT"

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.1.1

python main.py preprocess "$@"