import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from master import llm_name

import utils as u
from utils import CONSTS

BASE_PATH = os.path.join(u.get_root_dir(), "ros_results/")


def gen_ros_heatmap_data(name, file, llm, is_hybrid, is_static, episode):
    print(f"Plotting heatmap for {'static' if is_static else 'dynamic'} {name}")

    data = pd.read_csv(
        os.path.join(BASE_PATH, file),
        header=0,
        names=[
            "episode",
            "alice_start",
            "alice_end",
            "alice_goal",
            "bob_start",
            "bob_end",
            "bob_goal",
            "performance",
            "scenario",
            "env_change_rate",
        ],
    )

    data = data[data["episode"] == episode]

    # Split 'alice_end' and 'bob_end' columns
    data[["alice_end_x", "alice_end_y"]] = (
        data["alice_end"].str.strip("()").str.split(",", expand=True)
    )
    data[["bob_end_x", "bob_end_y"]] = data["bob_end"].str.strip("()").str.split(",", expand=True)

    # Drop the original 'alice_end' and 'bob_end' columns
    data = data.drop(
        [
            "alice_end",
            "alice_start",
            "alice_goal",
            "bob_end",
            "bob_start",
            "bob_goal",
            "scenario",
            "env_change_rate",
        ],
        axis=1,
    )

    # Convert the new columns to numeric type
    data[["alice_end_x", "alice_end_y", "bob_end_x", "bob_end_y"]] = data[
        ["alice_end_x", "alice_end_y", "bob_end_x", "bob_end_y"]
    ].astype(float)

    # Calculate total number of data points
    total_points = len(data)

    # Calculate proportions for Alice and Bob
    alice_proportions = (
        data.groupby(["alice_end_x", "alice_end_y"]).size().reset_index(name="alice_proportion")
    )
    alice_proportions["alice_proportion"] /= total_points
    alice_proportions["alice_proportion"] = alice_proportions["alice_proportion"].round(2)

    bob_proportions = (
        data.groupby(["bob_end_x", "bob_end_y"]).size().reset_index(name="bob_proportion")
    )
    bob_proportions["bob_proportion"] /= total_points
    bob_proportions["bob_proportion"] = bob_proportions["bob_proportion"].round(2)

    # Rename columns
    alice_proportions = alice_proportions.rename(columns={"alice_end_x": "x", "alice_end_y": "y"})
    bob_proportions = bob_proportions.rename(columns={"bob_end_x": "x", "bob_end_y": "y"})

    # Merge data
    data = pd.merge(alice_proportions, bob_proportions, on=["x", "y"], how="outer")

    # Fill NaN values with 0
    data["alice_proportion"] = data["alice_proportion"].fillna(0)
    data["bob_proportion"] = data["bob_proportion"].fillna(0)

    # Sort data
    data = data.sort_values(["x", "y"]).reset_index(drop=True)

    print(data)
    return data


def gen_ros_heatmap(heatmap_data, nsets=3):
    ndata = len(heatmap_data)  # Total number of heatmaps
    nheatmaps_per_set = 2  # Number of heatmaps in each cluster
    ngaps = nsets - 1  # Number of gaps between the clusters

    # Define the width ratios for the grid
    # Each heatmap takes up a ratio of 1, and each gap between clusters takes up a ratio of 0.5
    # width_ratios = [1] * ncols
    width_ratios = [1, 1, 0.5, 1, 1, 0.5, 1, 1]
    ncols = ndata + ngaps
    # for i in range(ngaps):
    #     width_ratios.insert((i + 1) * nheatmaps_per_set, 0.5)  # Insert a gap after each set

    fig = plt.figure(figsize=(CONSTS["WIDTH"] * 2, CONSTS["HEIGHT"]))

    # Create a gridspec with the width ratios and shared colorbars
    gs = fig.add_gridspec(
        3,
        ncols,
        width_ratios=width_ratios,
        wspace=0.1,
        hspace=0.5,
        height_ratios=[20, 1, 1],
    )

    alice_vmin = min(data[0]["alice_proportion"].min() for data in heatmap_data)
    alice_vmax = max(data[0]["alice_proportion"].max() for data in heatmap_data)
    bob_vmin = min(data[0]["bob_proportion"].min() for data in heatmap_data)
    bob_vmax = max(data[0]["bob_proportion"].max() for data in heatmap_data)

    print(heatmap_data)

    for idx, (data, is_hybrid, episode) in enumerate(heatmap_data):
        sns.set_theme(
            style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
        )

        alice_matrix = data.pivot(index="y", columns="x", values="alice_proportion").fillna(0)
        bob_matrix = data.pivot(index="y", columns="x", values="bob_proportion").fillna(0)

        full_index = pd.Index(range(8))
        full_columns = pd.Index(range(3))
        alice_matrix = alice_matrix.reindex(index=full_index, columns=full_columns, fill_value=0)
        bob_matrix = bob_matrix.reindex(index=full_index, columns=full_columns, fill_value=0)

        gaps = idx // nheatmaps_per_set
        col = idx + gaps
        ax = fig.add_subplot(gs[0, idx + gaps])

        sns.heatmap(
            alice_matrix,
            annot=True,
            fmt="",
            cmap="Reds",
            cbar=False,
            ax=ax,
            annot_kws={"size": CONSTS["FONT_SIZE"]},
            mask=(alice_matrix == 0),
            vmin=alice_vmin,
            vmax=alice_vmax,
        )

        sns.heatmap(
            bob_matrix,
            annot=True,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": CONSTS["FONT_SIZE"]},
            mask=(bob_matrix == 0),
            vmin=bob_vmin,
            vmax=bob_vmax,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            f"{'MARLIN' if is_hybrid else 'MARL'}/{episode}", fontsize=CONSTS["FONT_SIZE"]
        )

        ax.set_yticks(np.arange(8) + 0.5)
        ax.set_xticks(np.arange(3) + 0.5)
        ax.set_xticklabels([0, 1, 2])

        if col in [1, 4, 7]:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7])

        ax.tick_params(axis="both", which="major", labelsize=CONSTS["FONT_SIZE"])
        ax.invert_yaxis()

    # Add common color bars
    cax1 = fig.add_subplot(gs[1, :])
    cax2 = fig.add_subplot(gs[2, :])

    acb = plt.colorbar(ax.collections[0], cax=cax1, orientation="horizontal")
    bcb = plt.colorbar(ax.collections[1], cax=cax2, orientation="horizontal")

    acb.set_label(label="Alice End Proportion", size=CONSTS["FONT_SIZE"])
    bcb.set_label(label="Bob End Proportion", size=CONSTS["FONT_SIZE"])

    cax1.tick_params(labelsize=CONSTS["FONT_SIZE"])
    cax2.tick_params(labelsize=CONSTS["FONT_SIZE"])

    plt.tight_layout()
    # u.save_plot(fig, "multiple_heatmaps_row.pdf")
    plt.show()


# def gen_ros_heatmap(name, llm, data, ax=None):
#     shared_ax = False if ax is None else True

#     sns.set_theme(
#         style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
#     )

#     alice_matrix = data.pivot(index="y", columns="x", values="alice_count").fillna(0)
#     bob_matrix = data.pivot(index="y", columns="x", values="bob_count").fillna(0)

#     full_index = pd.Index(range(8))
#     full_columns = pd.Index(range(3))
#     alice_matrix = alice_matrix.reindex(index=full_index, columns=full_columns, fill_value=0)
#     bob_matrix = bob_matrix.reindex(index=full_index, columns=full_columns, fill_value=0)

#     # if not shared_ax:
#     #     fig, ax = plt.subplots(figsize=(CONSTS["WIDTH"] / 3, CONSTS["HEIGHT"]))

#     if not shared_ax:
#         fig, (ax, cax1, cax2) = plt.subplots(
#             nrows=3,
#             figsize=(CONSTS["WIDTH"] / 3, CONSTS["HEIGHT"] * 1.2),
#             gridspec_kw={"height_ratios": [20, 1, 1], "hspace": 0.3},
#         )

#     alice_heatmap = sns.heatmap(
#         alice_matrix,
#         annot=True,
#         fmt="",
#         cmap="Reds",
#         # alpha=0.5,
#         cbar=True,
#         cbar_ax=cax1,
#         ax=ax,
#         annot_kws={"size": CONSTS["FONT_SIZE"]},
#         mask=(alice_matrix == 0),
#         cbar_kws={"orientation": "horizontal"},
#     )

#     bob_heatmap = sns.heatmap(
#         bob_matrix,
#         annot=True,
#         fmt="",
#         cmap="Blues",
#         # alpha=0.5,
#         cbar=True,
#         cbar_ax=cax2,
#         ax=ax,
#         annot_kws={"size": CONSTS["FONT_SIZE"]},
#         mask=(bob_matrix == 0),
#         cbar_kws={"orientation": "horizontal"},
#     )

#     ax.set_xlabel("")
#     ax.set_ylabel("")

#     cax1.yaxis.set_label_position("left")
#     cax1.set_ylabel("Alice Count", fontsize=CONSTS["FONT_SIZE"], rotation=0, labelpad=25)
#     cax2.yaxis.set_label_position("left")
#     cax2.set_ylabel("Bob Count", fontsize=CONSTS["FONT_SIZE"], rotation=0, labelpad=25)

#     cax1.tick_params(labelsize=CONSTS["FONT_SIZE"])
#     cax2.tick_params(labelsize=CONSTS["FONT_SIZE"])

#     ax.set_xticks(np.arange(3) + 0.5)
#     ax.set_yticks(np.arange(8) + 0.5)
#     ax.set_xticklabels([0, 1, 2])
#     ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7])

#     ax.tick_params(axis="both", which="major", labelsize=CONSTS["FONT_SIZE"])

#     ax.invert_yaxis()

#     plt.tight_layout()

#     if not shared_ax:
#         u.save_plot(fig, f"g_real_heatmap_{name.replace(' ', '-')}.pdf")

#     # plt.show()


if __name__ == "__main__":
    heatmap_data = []
    for episode in [100, 850, 1600]:
        for is_hybrid in [True, False]:
            file = "hybrid_ros_results.csv" if is_hybrid else "marl_ros_results.csv"
            d = gen_ros_heatmap_data(
                "Maze Like Corridor", file, llm_name, is_hybrid, True, episode
            )
            heatmap_data.append((d, is_hybrid, episode))

    print(heatmap_data)
    gen_ros_heatmap(heatmap_data)
