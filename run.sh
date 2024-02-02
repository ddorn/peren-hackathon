#!/bin/bash

# run.sh: (mandatory). It will be called with a path to the questions csv file and with a path to an output directory.
# It should set-up your environment, such as Jean-Zay module loading and python environment sourcing, then run your program.
# Your program should read each entry of the csv, then output the corresponding the corresponding answer, the prompt used to generate such answer and url corresponding to the document used to generate the answer.

set -e

# HuggingFace
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Unstructured
export SCARF_NO_ANALYTICS=true
# Chroma db
export ANONYMIZED_TELEMETRY=False

QUESTIONS=$(realpath $1)
OUTPUT=$(realpath $2)

ROOT=$(dirname $0)
cd $ROOT

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.1.1

python main.py run "$QUESTIONS" "$OUTPUT"
