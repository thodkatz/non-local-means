# %%
import re
from datetime import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv(
    "../results/results.csv",
)

df["speedup"] = 1
df["dynamic_size"] = 1

for i in df.index:
    df.loc[i, "dataset"] = df.loc[i, "dataset"].split("/")[-1].split("_")[2]
    df.loc[i, "version"] = df.loc[i, "version"].split("/")[-1]
    df.loc[i, "dynamic_size"] = 2 * df.loc[i, "patch_size"] ** 2 * df.loc[i, "threads"] * 4

print(df)

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300


# %%
df = df[(df.threads <= 1024) & (df.blocks <= 1024) & (df.blocks != 32)]
v1 = df[df["version"] == "v1"]
# v3 = df[df["version"] == "v3"].loc[:, ["dataset", "library", "threads", "time", "speedup"]]
v2 = df[(df["version"] == "v2") & (df.dynamic_size <= 48384)]
# v4 = df[df["version"] == "v4"].loc[:, ["dataset", "library", "threads", "time", "speedup"]]

versions = [v1, v2]
# versions = [v1]

for v in versions:

    datasets = v.dataset.unique()
    threads = v.threads.unique()
    blocks = v.blocks.unique()
    patch_sizes = v.patch_size.unique()
    version = v.version.unique()[0]
    print(version)

    # Calculate Speedup
    for patch_size in patch_sizes:
        for dataset_name in datasets:
            d = v[(v.dataset == dataset_name) & (v.patch_size == patch_size)]
            baseline_time = d.time.max()
            for i in d.index:
                v.loc[i, "speedup"] = baseline_time / d.loc[i].time

        with sns.axes_style("whitegrid"):
            g = sns.catplot(
                data=v[
                    (v.patch_size == patch_size)
                    & (
                        (v.threads == 32)
                        | (v.threads == 64)
                        | (v.threads == 96)
                        | (v.threads == 128)
                        | (v.threads == 256)
                        | (v.threads == 512)
                        | (v.threads == 1024)
                    )
                ],
                x="threads",
                hue="blocks",
                y="speedup",
                palette="Set2",
                kind="bar",
                col="dataset",
                ci=95,
                # alpha=0.8,
            )
            g.savefig(f"../results/figures/{version}_{patch_size}.pdf")


# %%
for v in [v1, v2]:
    version = v.version.unique()[0]
    # Get best speedup for each patch_size and dataset
    for patch_size in patch_sizes:
        for dataset_name in datasets:
            d = v[(v.dataset == dataset_name) & (v.patch_size == patch_size)]
            min_time = d.time.min()
            zeugos = d[d.time == min_time].iloc[0]
            print(
                f"Version: {version}\tDataset: {dataset_name}\tPatch_size: {patch_size}\tBest speedup: {round(min_time,4)}\tZeugos: {int(zeugos.blocks)} - {int(zeugos.threads)}"
            )

    # with sns.axes_style("whitegrid"):

    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     ax.set(
    #         xlabel="Threads (n)",
    #         ylabel="Speedup (x)",
    #         title=f"{version} using: cuda",
    #         # xscale="log",
    #     )

    #     # ax.set_xscale("log")
    #     ax.set_xticklabels(blocks, rotation=45)
    #     ax.set_xticks(threads)

    #     # ax.set_ylim(0, 16)

    #     # ax.set_xlim((min(threads), max(threads)))
    #     # ax.set_xticks(np.arange(0, max(threads) + 1, 1))
    #     # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: x if x in threads else None))

    #     sns.lineplot(
    #         data=v[
    #             (v.dataset == "lena")
    #             # & ((v.threads == 32) | (v.threads == 128) | (v.threads == 512) | (v.threads == 1024))
    #         ],
    #         x="threads",
    #         y="speedup",
    #         hue="blocks",
    #         style="dataset",
    #         palette="Set2",
    #         markers=True,
    #         markersize=5,
    #         linewidth=1.5,
    #         ci=0,
    #         # err_style="bars",
    #         # marker="o",
    #     )

    #     # sns.barplot(
    #     #     data=v[
    #     #         (v.dataset == "lena")
    #     #         & ((v.threads == 32) | (v.threads == 128) | (v.threads == 512) | (v.threads == 1024))
    #     #     ],
    #     #     x="threads",
    #     #     y="speedup",
    #     #     hue="blocks",
    #     #     # style="dataset",
    #     #     palette="Set2",
    #     #     # markers=True,
    #     #     # markersize=5,
    #     #     linewidth=1.5,
    #     #     ci=0,
    #     #     # err_style="bars",
    #     #     # marker="o",
    #     # )

    #     # ax2 = ax.twinx()
    #     # # ax2.plot(100 * np.random.rand(10))
    #     # sns.lineplot(
    #     #     data=v[v.library == library],
    #     #     x="threads",
    #     #     y="time",
    #     #     hue="dataset",
    #     #     style="dataset",
    #     #     palette="Set2",
    #     #     markers=True,
    #     #     markersize=10,
    #     #     linewidth=2.5,
    #     #     ci=95,
    #     #     # err_style="bars",
    #     #     # marker="o",
    #     # )

    #     fig.savefig(f"../results/figures/{version}.pdf")
    #     # fig.legend(
    #     #     labels=[
    #     #         "test",
    #     #     ]
    #     # )
