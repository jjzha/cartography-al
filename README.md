# Cartography Active Learning

**\[Important note 31/01/2021\]**: The code currently consists of a bug that needs resolving. We plan to address this bug in the coming months.

This repository contains the code and data for the paper:

Mike Zhang and Barbara Plank. 2021. [**Cartography Active Learning**](https://aclanthology.org/2021.findings-emnlp.36.pdf). In Findings of the Association for Computational Linguistics: EMNLP 2021.

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
```
@inproceedings{zhang-plank-2021-cartography-active,
    title = "Cartography Active Learning",
    author = "Zhang, Mike  and
      Plank, Barbara",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.36",
    pages = "395--406",
    abstract = "We propose Cartography Active Learning (CAL), a novel Active Learning (AL) algorithm that exploits the behavior of the model on individual instances during training as a proxy to find the most informative instances for labeling. CAL is inspired by data maps, which were recently proposed to derive insights into dataset quality (Swayamdipta et al., 2020). We compare our method on popular text classification tasks to commonly used AL strategies, which instead rely on post-training behavior. We demonstrate that CAL is competitive to other common AL methods, showing that training dynamics derived from small seed data can be successfully used for AL. We provide insights into our new AL method by analyzing batch-level statistics utilizing the data maps. Our results further show that CAL results in a more data-efficient learning strategy, achieving comparable or better results with considerably less training data.",
}
```
## Contact
If there is any issue, please reach out to Mike Zhang (mikz@itu.dk) or create an issue in this repository.
  
