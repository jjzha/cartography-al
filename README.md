# Cartography Active Learning

This repository contains the code and data for the paper:

Mike Zhang and Barbara Plank. 2021. **Cartography Active Learning**. To appear in Findings of the Association for Computational Linguistics: EMNLP 2021.

## Repository
In this repository you will find:

* `project/src/*`: all the code for the experiments.
* `project/resources/data/*`: the data used in our paper.
* `run.sh`: all commands necessary to rerun the experiments
in the paper.
* `requirements.txt`: all packages necessary for reproducibility.
* `.env`: all environment variables. 
  
**Important Note**: if you don't want to run all the scripts sequentially, at
least use the command 

`mkdir -p project/{resources/{cartography_plots,embeddings,indices,mapping},results/{agnews,trec},plots/{agnews,trec}}
` 

to make sure that there are folders available for the files to go into.

## Citation
TBD

## Contact
If there is any issue, please reach out to Mike Zhang (mikz@itu.dk) or create an issue in this repository.
  