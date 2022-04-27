#!/bin/bash

# Prepare folders for files to go into
mkdir -p project/{resources/{cartography_plots,embeddings,indices,mapping,logs/{agnews,trec}},results/{agnews,trec},plots/{agnews,trec}}

# Run all acquisition functions for TREC
# run_trec.sh

# Run all acquisition functions for AGNEWS
# run_agnews.sh

# Plots results in a lineplot (Figure 3) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --initial_size 500 --plot_results
python3 main.py --task agnews --initial_size 1000 --plot_results

# Run significant tests (Table 2) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --significance
python3 main.py --task agnews --significance

# Run overlapping indices (Table 4) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --check_indices
python3 main.py --task agnews --check_indices
