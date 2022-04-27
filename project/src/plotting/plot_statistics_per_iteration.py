import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    confidence_agnews = {
            "random": [0.51, 0.577, 0.68, 0.713, 0.715, 0.745, 0.766, 0.761, 0.739, 0.762, 0.814, 0.713, 0.798, 0.753,
                       0.849, 0.783, 0.831, 0.752, 0.711, 0.777, 0.748, 0.761, 0.691, 0.74, 0.787, 0.733, 0.726, 0.643,
                       0.718, 0.727],
            "leastconfidence": [0.381, 0.352, 0.319, 0.334, 0.342, 0.36, 0.345, 0.327, 0.382, 0.344, 0.351, 0.356,
                                0.386, 0.413, 0.391, 0.355, 0.412, 0.419, 0.385, 0.379, 0.368, 0.527, 0.464, 0.42,
                                0.377, 0.416, 0.439, 0.473, 0.383, 0.501],
            "discriminative": [0.695, 0.788, 0.846, 0.913, 0.884, 0.847, 0.877, 0.939, 0.898, 0.89, 0.894, 0.923, 0.89,
                               0.901, 0.877, 0.826, 0.883, 0.762, 0.901, 0.886, 0.927, 0.83, 0.872, 0.965, 0.938, 0.86,
                               0.882, 0.907, 0.893, 0.853],
            "cartography": [0.671, 0.698, 0.667, 0.64, 0.573, 0.501, 0.535, 0.55, 0.558, 0.515, 0.48, 0.54, 0.446,
                            0.476, 0.544, 0.462, 0.526, 0.487, 0.611, 0.526, 0.421, 0.526, 0.528, 0.582, 0.564, 0.545,
                            0.62, 0.547, 0.625, 0.607]}

    variability_agnews = {
            "random": [0.126, 0.112, 0.112, 0.09, 0.101, 0.101, 0.085, 0.102, 0.077, 0.098, 0.077, 0.09, 0.09, 0.097,
                       0.075, 0.106, 0.082, 0.1, 0.089, 0.1, 0.099, 0.111, 0.119, 0.124, 0.119, 0.104, 0.124, 0.136,
                       0.128, 0.137],
            "leastconfidence": [0.125, 0.128, 0.114, 0.131, 0.128, 0.135, 0.141, 0.13, 0.137, 0.131, 0.15, 0.139, 0.153,
                                0.149, 0.157, 0.142, 0.155, 0.168, 0.18, 0.165, 0.151, 0.159, 0.169, 0.162, 0.14, 0.147,
                                0.168, 0.169, 0.156, 0.175],
            "discriminative": [0.132, 0.099, 0.083, 0.068, 0.069, 0.094, 0.068, 0.043, 0.034, 0.054, 0.052, 0.047, 0.05,
                               0.065, 0.079, 0.084, 0.065, 0.092, 0.054, 0.06, 0.048, 0.069, 0.09, 0.033, 0.04, 0.089,
                               0.08, 0.063, 0.071, 0.069],
            "cartography": [0.108, 0.105, 0.096, 0.118, 0.123, 0.115, 0.135, 0.114, 0.12, 0.141, 0.15, 0.141, 0.136,
                            0.149, 0.135, 0.136, 0.153, 0.152, 0.139, 0.158, 0.148, 0.141, 0.158, 0.145, 0.15, 0.139,
                            0.161, 0.151, 0.147, 0.151]
            }
    correctness_agnews = {
            "random": [0.66, 0.704, 0.792, 0.804, 0.8, 0.828, 0.86, 0.832, 0.788, 0.828, 0.86, 0.744, 0.852, 0.796,
                       0.904, 0.844, 0.864, 0.788, 0.752, 0.84, 0.796, 0.812, 0.748, 0.808, 0.872, 0.788, 0.784, 0.716,
                       0.8, 0.788],
            "leastconfidence": [0.532, 0.436, 0.38, 0.412, 0.44, 0.452, 0.412, 0.392, 0.484, 0.44, 0.46, 0.444, 0.492,
                                0.552, 0.492, 0.436, 0.504, 0.5, 0.444, 0.436, 0.432, 0.632, 0.544, 0.504, 0.44, 0.504,
                                0.52, 0.564, 0.432, 0.588],
            "discriminative": [0.792, 0.856, 0.876, 0.948, 0.92, 0.888, 0.904, 0.96, 0.908, 0.912, 0.916, 0.948, 0.896,
                               0.94, 0.92, 0.864, 0.892, 0.792, 0.928, 0.9, 0.932, 0.852, 0.912, 0.972, 0.948, 0.916,
                               0.912, 0.94, 0.952, 0.892],
            "cartography": [0.808, 0.86, 0.784, 0.764, 0.7, 0.556, 0.696, 0.684, 0.684, 0.628, 0.588, 0.684, 0.536,
                            0.548, 0.68, 0.512, 0.604, 0.572, 0.68, 0.62, 0.488, 0.644, 0.66, 0.676, 0.636, 0.62, 0.756,
                            0.648, 0.716, 0.736]
            }

    confidence_trec = {
            "random": [0.27, 0.284, 0.423, 0.431, 0.396, 0.445, 0.433, 0.552, 0.432, 0.533, 0.531, 0.527, 0.558, 0.589,
                       0.449, 0.548, 0.585, 0.575, 0.595, 0.529, 0.568, 0.511, 0.609, 0.539, 0.603, 0.576, 0.576, 0.54,
                       0.541, 0.525],
            "leastconfidence": [0.216, 0.244, 0.295, 0.271, 0.302, 0.278, 0.267, 0.331, 0.316, 0.304, 0.382, 0.332,
                                0.348, 0.326, 0.36, 0.368, 0.384, 0.345, 0.37, 0.403, 0.449, 0.378, 0.399, 0.391, 0.385,
                                0.437, 0.479, 0.449, 0.473, 0.449],
            "discriminative": [0.257, 0.352, 0.434, 0.551, 0.519, 0.556, 0.578, 0.606, 0.583, 0.621, 0.582, 0.585,
                               0.579, 0.578, 0.742, 0.693, 0.507, 0.64, 0.545, 0.54, 0.536, 0.518, 0.591, 0.542, 0.555,
                               0.483, 0.441, 0.435, 0.51, 0.539],
            "cartography": [0.244, 0.354, 0.384, 0.403, 0.405, 0.427, 0.359, 0.342, 0.371, 0.359, 0.392, 0.333, 0.414,
                            0.387, 0.418, 0.439, 0.504, 0.471, 0.426, 0.42, 0.496, 0.434, 0.501, 0.469, 0.497, 0.44,
                            0.436, 0.485, 0.543, 0.478]
            }
    variability_trec = {
            "random": [0.065, 0.075, 0.093, 0.091, 0.101, 0.115, 0.12, 0.096, 0.114, 0.118, 0.118, 0.122, 0.141, 0.131,
                       0.148, 0.13, 0.15, 0.159, 0.152, 0.163, 0.171, 0.166, 0.166, 0.192, 0.176, 0.164, 0.172, 0.192,
                       0.157, 0.184],
            "leastconfidence": [0.046, 0.063, 0.095, 0.095, 0.112, 0.107, 0.103, 0.135, 0.124, 0.129, 0.147, 0.14,
                                0.135, 0.145, 0.151, 0.149, 0.186, 0.155, 0.175, 0.153, 0.169, 0.17, 0.167, 0.172,
                                0.153, 0.183, 0.179, 0.179, 0.188, 0.188],
            "discriminative": [0.063, 0.09, 0.117, 0.105, 0.084, 0.124, 0.148, 0.112, 0.123, 0.124, 0.141, 0.139, 0.158,
                               0.136, 0.087, 0.165, 0.182, 0.134, 0.169, 0.16, 0.155, 0.198, 0.196, 0.154, 0.172, 0.175,
                               0.174, 0.169, 0.174, 0.187],
            "cartography": [0.062, 0.085, 0.087, 0.088, 0.097, 0.108, 0.09, 0.102, 0.114, 0.12, 0.125, 0.118, 0.153,
                            0.139, 0.151, 0.165, 0.166, 0.166, 0.158, 0.178, 0.163, 0.169, 0.183, 0.161, 0.172, 0.179,
                            0.161, 0.169, 0.166, 0.203]
            }
    correctness_trec = {
            "random": [0.432, 0.404, 0.572, 0.528, 0.456, 0.544, 0.54, 0.624, 0.512, 0.644, 0.652, 0.612, 0.656, 0.704,
                       0.52, 0.608, 0.684, 0.676, 0.684, 0.616, 0.628, 0.592, 0.688, 0.644, 0.696, 0.68, 0.648, 0.596,
                       0.6, 0.592],
            "leastconfidence": [0.28, 0.34, 0.436, 0.36, 0.436, 0.36, 0.344, 0.432, 0.42, 0.352, 0.488, 0.38, 0.436,
                                0.384, 0.436, 0.436, 0.468, 0.424, 0.468, 0.5, 0.54, 0.428, 0.484, 0.452, 0.524, 0.54,
                                0.592, 0.552, 0.576, 0.536],
            "discriminative": [0.424, 0.556, 0.608, 0.64, 0.536, 0.632, 0.66, 0.724, 0.668, 0.696, 0.696, 0.696, 0.688,
                               0.644, 0.788, 0.768, 0.58, 0.684, 0.624, 0.612, 0.604, 0.576, 0.644, 0.668, 0.648, 0.544,
                               0.532, 0.516, 0.6, 0.636],
            "cartography": [0.388, 0.516, 0.5, 0.524, 0.476, 0.58, 0.46, 0.432, 0.544, 0.448, 0.512, 0.436, 0.536,
                            0.476, 0.528, 0.532, 0.632, 0.584, 0.5, 0.512, 0.612, 0.484, 0.64, 0.568, 0.628, 0.504,
                            0.532, 0.584, 0.652, 0.596]
            }

    sns.set(style="whitegrid")
    paper_rc = {'lines.linewidth': 1.7, 'lines.markersize': 4.5}
    sns.set_context("paper", rc=paper_rc, font_scale=1.1)
    pal = sns.diverging_palette(260, 15, n=4, sep=10, center="dark")
    markers = {"random": "P", "leastconfidence": "^", "discriminative": "X", "cartography": "o"}

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    # confidence agnews
    df = pd.DataFrame.from_dict(confidence_agnews)
    df.index.name = "iteration"
    df = df.stack().reset_index()
    sns.lineplot(data=df, x="iteration", y=0, hue="level_1", style="level_1",
                 palette=pal, dashes=False, markers=markers, ax=ax[0][0])
    ax[0][0].title.set_text("AGNews - Confidence")
    ax[0][0].set_xlabel("Iteration")
    ax[0][0].set_ylabel("Confidence")

    # variability agnews
    df2 = pd.DataFrame.from_dict(variability_agnews)
    df2.index.name = "iteration"
    df2 = df2.stack().reset_index()
    sns.lineplot(data=df2, x="iteration", y=0, hue="level_1", style="level_1",
                 palette=pal, dashes=False, markers=markers, ax=ax[0][1])
    ax[0][1].title.set_text("AGNews - Variability")
    ax[0][1].set_xlabel("Iteration")
    ax[0][1].set_ylabel("Variability")

    # correctness agnews
    df3 = pd.DataFrame.from_dict(correctness_agnews)
    df3.index.name = "iteration"
    df3 = df3.stack().reset_index()
    sns.lineplot(data=df3, x="iteration", y=0, hue="level_1", style="level_1",
                 palette=pal, dashes=False, markers=markers, ax=ax[0][2])
    ax[0][2].title.set_text("AGNews - Correctness")
    ax[0][2].set_xlabel("Iteration")
    ax[0][2].set_ylabel("Correctness")

    # confidence trec
    df4 = pd.DataFrame.from_dict(confidence_trec)
    df4.index.name = "iteration"
    df4 = df4.stack().reset_index()
    sns.lineplot(data=df4, x="iteration", y=0, hue="level_1", style="level_1",
                 palette=pal, dashes=False, markers=markers, ax=ax[1][0])
    ax[1][0].title.set_text("TREC - Confidence")
    ax[1][0].set_xlabel("Iteration")
    ax[1][0].set_ylabel("Confidence")

    # variability trec
    df5 = pd.DataFrame.from_dict(variability_trec)
    df5.index.name = "iteration"
    df5 = df5.stack().reset_index()
    sns.lineplot(data=df5, x="iteration", y=0, hue="level_1", style="level_1",
                 palette=pal, dashes=False, markers=markers, ax=ax[1][1])
    ax[1][1].title.set_text("TREC - Variability")
    ax[1][1].set_xlabel("Iteration")
    ax[1][1].set_ylabel("Variability")

    # correctness trec
    df6 = pd.DataFrame.from_dict(correctness_trec)
    df6.index.name = "iteration"
    df6 = df6.stack().reset_index()
    sns.lineplot(data=df6, x="iteration", y=0, hue="level_1", style="level_1",
                 palette=pal, dashes=False, markers=markers, ax=ax[1][2])
    ax[1][2].title.set_text("TREC - Correctness")
    ax[1][2].set_xlabel("Iteration")
    ax[1][2].set_ylabel("Correctness")

    for row in ax:
        for item in row:
            item.get_legend().remove()

    plt.legend(fancybox=True, shadow=True, title="Sampling strategy", loc="upper left", bbox_to_anchor=(1.0, 1.0),
               ncol=1, prop={'size': 10})

    plt.tight_layout()
    plt.savefig("plot_per_iteration.pdf", dpi=300)
