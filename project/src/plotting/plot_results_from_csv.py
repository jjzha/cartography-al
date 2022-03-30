import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_from_csv(args: argparse.Namespace):
    df_dict = {"accuracy": [], "interval": [], "strategy": []}
    strategies = ["bald", "cartography", "discriminative", "entropy", "leastconfidence", "random"]

    if args.task == "agnews":
        total_size = float(os.getenv("MAX_INSTANCE_AGNEWS"))
    else:
        total_size = float(os.getenv("MAX_INSTANCE_TREC"))

    for strat in strategies:
        for entry in os.scandir(f"{os.getenv('RESULTS_PATH')}{args.task}"):
            strategy = entry.path.split("/")[-1].split("_")[0]
            if entry.path.endswith(".csv") and strategy == strat:

                with open(entry.path) as f:
                    df = pd.read_csv(f, sep="\t")
                    df_dict["accuracy"] += df["score"].tolist()
                    df_dict["interval"] += [(s + float(args.initial_size)) / total_size * 100 for s in df["step"].tolist()]
                    df_dict["strategy"] += [strategy for _ in range(len(df))]

    sns.set(style="whitegrid")
    paper_rc = {'lines.linewidth': 1.8, 'lines.markersize': 5}
    sns.set_context("paper", rc=paper_rc, font_scale=1.1)
    pal = sns.diverging_palette(260, 15, n=6, sep=10, center="dark")
    markers = {"random"     : "P", "entropy": "s", "leastconfidence": "^", "bald": "d", "discriminative": "X",
               "cartography": "o"}
    ax = sns.lineplot(data=df_dict,
                      x="interval",
                      y="accuracy",
                      hue="strategy",
                      style="strategy",
                      style_order=["random", "entropy", "leastconfidence", "bald", "discriminative", "cartography"],
                      hue_order=["random", "entropy", "leastconfidence", "bald", "discriminative", "cartography"],
                      markers=markers,
                      palette=pal,
                      ci=None)
    ax.set(xlabel="Percentage of Data Used", ylabel="Accuracy",
           title=f"Dataset: {args.task.upper()}, Seed set size: {args.initial_size}")
    ax.legend(fancybox=True, shadow=True, title="Sampling strategy", loc="lower right", bbox_to_anchor=(1.0, 0.0),
              ncol=1)
    plt.tight_layout()
    plt.savefig(f"{os.getenv('PLOT_PATH')}{args.task}/{args.task}_results_{args.initial_size}.pdf", dpi=300)
