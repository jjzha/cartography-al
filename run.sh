#!/bin/bash

# Prepare folders for files to go into
mkdir -p project/{resources/{cartography_plots,embeddings,indices,mapping},results/{agnews,trec},plots/{agnews,trec}}

# Get Data Maps
# Figure 1 -- if you want them annotated -> project/src/utils/save_cartography.py
python3 main.py --task trec --initial_size 5452 --batch_size 16 --pretrained --freeze --cartography --plot
python3 main.py --task agnews --initial_size 120000 --batch_size 64 --pretrained --freeze --cartography --plot
# Figure 2 -- if you want them annotated -> project/src/utils/save_cartography.py
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --cartography --plot
python3 main.py --task agnews --initial_size 1000 --batch_size 64 --pretrained --freeze --cartography --plot
# Full data maps (Appendix)
python3 main.py --task trec --initial_size 5452 --batch_size 16 --pretrained --freeze --cartography --plot --histogram
python3 main.py --task agnews --initial_size 120000 --batch_size 64 --pretrained --freeze --cartography --plot --histogram

# Run all acquisition functions for TREC
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition random --analysis
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition entropy
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition leastconfidence --analysis
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition bald
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition discriminative --analysis
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition cartography --analysis

# Run all acquisition functions for AGNEWS
python3 main.py --task agnews --initial_size 1000 --batch_size 64 --pretrained --freeze --acquisition random --analysis
python3 main.py --task agnews --initial_size 1000 --batch_size 64 --pretrained --freeze --acquisition entropy
python3 main.py --task agnews --initial_size 1000 --batch_size 64 --pretrained --freeze --acquisition leastconfidence --analysis
python3 main.py --task agnews --initial_size 1000 --batch_size 64 --pretrained --freeze --acquisition bald
python3 main.py --task agnews --initial_size 1000 --batch_size 64 --pretrained --freeze --acquisition discriminative --analysis
python3 main.py --task agnews --initial_size 1000 --batch_size 64 --pretrained --freeze --acquisition cartography --analysis

# Plots results in a lineplot (Figure 3) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --initial_size 500 --plot_results
python3 main.py --task agnews --initial_size 1000 --plot_results

# Run significant tests (Table 2) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --significance
python3 main.py --task agnews --significance

# Run overlapping indices (Table 4) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --check_indices
python3 main.py --task agnews --check_indices
