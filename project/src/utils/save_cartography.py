import argparse
import json
import logging
import os
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def save_cartography(args: argparse.Namespace, cartography: Dict):
    logger.info(f"Saving mappings to: {os.getenv('MAPPING_PATH')}")
    cartography_mapping_path = f"{os.getenv('MAPPING_PATH')}cartography_{args.task}_{args.initial_size}_" \
                               f"{os.getenv('EPOCHS')}_freeze_{args.freeze}_by_idx.json"

    with open(cartography_mapping_path, "w", encoding="utf-8") as f:
        json.dump(cartography, f, indent=4)


def plot_cartography(args: argparse.Namespace, cartography: Dict) -> None:
    logger.info(f"Saving plot to: {os.getenv('CARTOGRAPHY_PATH')}")
    cartography_plot_path = f"{os.getenv('CARTOGRAPHY_PATH')}cartography_{args.task}_{args.initial_size}_" \
                            f"{os.getenv('EPOCHS')}_freeze_{args.freeze}_annotated.pdf"

    if args.initial_size == 500:
        pal = sns.diverging_palette(260, 15, n=6, sep=10, center="dark")  # change to n if wrong number of colors
    else:
        pal = sns.diverging_palette(260, 15, n=6, sep=10, center="dark")

    if args.histogram:
        cartography_plot_path = f"{os.getenv('CARTOGRAPHY_PATH')}cartography_{args.task}_{args.initial_size}_" \
                                f"{os.getenv('EPOCHS')}_freeze_{args.freeze}_histogram.png"
        sns.set(style="whitegrid", context="paper", font_scale=1.6)
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax = fig.add_subplot(gs[:, 0])

        sns.scatterplot(x=cartography["variability"],
                        y=cartography["confidence"],
                        hue=cartography["correctness"],
                        style=cartography["correctness"],
                        palette=pal,
                        ax=ax)
    else:
        sns.set(style="whitegrid", context="paper", rc={'figure.figsize': (8, 8)}, font_scale=1.6)
        ax = sns.scatterplot(x=cartography["variability"],
                             y=cartography["confidence"],
                             hue=cartography["correctness"],
                             style=cartography["correctness"],
                             palette=pal)

        # For annotating the regions uncomment the commented lines below

        # bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
        # func_annotate = lambda text, xyc, bbc: ax.annotate(text,
        #                                                    xy=xyc,
        #                                                    xycoords="axes fraction",
        #                                                    fontsize=15,
        #                                                    color='black',
        #                                                    va="center",
        #                                                    ha="center",
        #                                                    rotation=350,
        #                                                    bbox=bb(bbc))
        # an1 = func_annotate("ambiguous", xyc=(0.7, 0.5), bbc='black')
        # an2 = func_annotate("easy-to-learn", xyc=(0.3, 0.8), bbc='r')
        # an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.20), bbc='b')

    ax.legend(fancybox=True, shadow=True, ncol=1)
    ax.set(xlabel="variability", ylabel="confidence", title=f"{args.task.upper()}-MLP Data Map")
    ax.get_legend().set_title("correct.")

    if args.histogram:
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = sns.histplot(cartography["confidence"], ax=ax1, color='#622a87')
        plott0.set_title('')
        plott0.set_xlabel('confidence')
        plott0.set_ylabel('density')
        plott1 = sns.histplot(cartography["variability"], ax=ax2, color='teal')
        plott1.set_title('')
        plott1.set_xlabel('variability')
        plott1.set_ylabel('density')
        plot2 = sns.countplot(cartography["correctness"], ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True)  # Show the vertical gridlines
        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

    plt.tight_layout()
    plt.savefig(cartography_plot_path, dpi=72)
