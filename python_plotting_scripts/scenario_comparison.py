import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils as u
from utils import CONSTS

BASE_PATH = os.path.join(u.get_root_dir(), "csvs_for_graphs/")


def gen_scenario_comparison_data(
    info,
    episode,
    llm,
    is_static,
    env_change_rate=None,
    env=None,
):
    print(f"Processing data for all scenarios at episode {episode}")

    # Load MARLIN data
    data_marlin = u.join_csvs(
        os.path.join(BASE_PATH, "hybrid/"),
        [
            f"{'s' if is_static else 'd'}_{value['hybrid_file']}_{'static_env' if is_static else f'dynamic_env_{env_change_rate}'}.csv"
            for value in info.values()
        ],
    )
    data_marlin["method"] = "MARLIN"

    # Load MARL data
    data_marl = u.join_csvs(
        os.path.join(BASE_PATH, "marl/"),
        [
            f"{'s' if is_static else 'd'}_{value['traditional_file']}_{'static_env' if is_static else f'dynamic_env_{env_change_rate}'}.csv"
            for value in info.values()
        ],
    )
    data_marl["method"] = "MARL"

    # Combine MARLIN and MARL data
    data = pd.concat([data_marlin, data_marl])

    # Filter for the specified episode
    data = data[data["episode"] == episode]

    if env is not None:
        env = env.replace(" ", "_")
        data = data[data["scenario"] == env]

    print(f"data {data}")

    # Calculate average performance
    data_avg = (
        data.groupby(["episode", "scenario"])
        .agg({"performance": ["median", "std", "count"]})
        .reset_index()
    )

    # Rename columns
    data_avg.columns = ["episode", "scenario", "performance_avg", "performance_sd", "n"]

    # Calculate standard error
    data_avg["performance_se"] = data_avg["performance_sd"] / np.sqrt(data_avg["n"])

    # Convert scenario to categorical and set the order
    data["scenario"] = pd.Categorical(
        data["scenario"], categories=data["scenario"].unique(), ordered=True
    )

    # Print data and statistics
    print(data)
    print(data_avg)
    print(data["performance"].min())
    print(data["performance"].max())
    print(f"Overall median: {data_avg['performance_avg'].median()}")

    return data


def _gen_scenario_comparison_single_env_single_ep(env, data, episode, ax=None):
    shared_ax = False if ax is None else True

    sns.set_theme(
        style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
    )

    # Create the plot
    if not shared_ax:
        fig, ax = plt.subplots(figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]))

    sns.boxplot(
        x="scenario",
        y="performance",
        hue="method",
        data=data,
        width=0.75,
        linewidth=0.5,
        fliersize=3,
        saturation=1,
        palette={"MARLIN": CONSTS["MARLIN_COLOUR"], "MARL": CONSTS["MARL_COLOUR"]},
        ax=ax,
    )

    # Customize the plot
    plt.ylim(0, 1)
    plt.xticks(
        range(len(data["scenario"].unique())),
        [
            s.replace("_", " ").replace(" Corridor", "\nCorridor")
            for s in data["scenario"].unique()
        ],
        rotation=0,
    )

    # Legend
    # if not shared_ax:
    # fig.subplots_adjust(bottom=0.2)

    plt.legend(
        loc="upper right",
        # frameon=False,
        fontsize=CONSTS["FONT_SIZE"],
    )

    # Remove axes and labels
    plt.ylabel("Performance", fontsize=CONSTS["FONT_SIZE"])
    plt.xlabel("", fontsize=CONSTS["FONT_SIZE"])
    plt.yticks([0, 0.25, 0.50, 0.75, 1.0], fontsize=CONSTS["FONT_SIZE"])
    plt.xticks(fontsize=CONSTS["FONT_SIZE"])

    if not shared_ax:
        plt.title(f"Episode {episode}", fontsize=CONSTS["FONT_SIZE"])

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    # plt.gca().spines["bottom"].set_visible(False)

    # Adjust layout and display
    plt.tight_layout()

    if not shared_ax:
        u.save_plot(
            fig,
            f"g_{episode}_scenario_comparison_sim_{env.replace(' ', '-') if env is not None else 'all_envs'}.pdf",
        )
        # plt.show()
    return ax


def _gen_scenario_comparison_multiple_env_single_ep(env, data, episode, ax=None):
    shared_ax = False if ax is None else True

    sns.set_theme(
        style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
    )

    # Create the plot
    if not shared_ax:
        fig, ax = plt.subplots(figsize=(CONSTS["WIDTH"] * 1.25, CONSTS["HEIGHT"]))

    sns.boxplot(
        x="scenario",
        y="performance",
        hue="method",
        data=data,
        width=0.75,
        linewidth=0.5,
        fliersize=3,
        saturation=1,
        palette={"MARLIN": CONSTS["MARLIN_COLOUR"], "MARL": CONSTS["MARL_COLOUR"]},
        ax=ax,
    )

    # Customize the plot
    plt.ylim(0, 1)
    plt.xticks(
        range(len(data["scenario"].unique())),
        [
            s.replace("_", "\n")  # .replace(" Corridor", "\nCorridor")
            for s in data["scenario"].unique()
        ],
        fontsize=CONSTS["FONT_SIZE"],
    )

    # ax.tick_params(axis="x", rotation=45)

    # Legend
    # if not shared_ax:
    # fig.subplots_adjust(bottom=0.2)

    plt.legend(
        title="Method",
        loc="upper right",
        fontsize=CONSTS["FONT_SIZE"],
    )

    # Remove axes and labels
    plt.ylabel("Performance", fontsize=CONSTS["FONT_SIZE"])
    plt.xlabel("", fontsize=CONSTS["FONT_SIZE"])
    plt.yticks([0, 0.25, 0.50, 0.75, 1.0], fontsize=CONSTS["FONT_SIZE"])
    plt.xticks(fontsize=CONSTS["FONT_SIZE"])

    if not shared_ax:
        plt.title(f"Episode {episode}", fontsize=CONSTS["FONT_SIZE"])

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["left"].set_visible(False)
    # plt.gca().spines["bottom"].set_visible(False)

    # Adjust layout and display
    plt.tight_layout()

    if not shared_ax:
        u.save_plot(
            fig,
            f"g_{episode}_scenario_comparison_sim_{env.replace(' ', '-') if env is not None else 'all_envs'}.pdf",
        )
        # plt.show()
    return ax


def _gen_scenario_comparison_single_ep(env, data, episode, ax=None):
    if env is not None:
        print("Plotting one env")
        return _gen_scenario_comparison_single_env_single_ep(env, data, episode, ax)
    else:
        print("Plotting multiple envs")
        return _gen_scenario_comparison_multiple_env_single_ep(env, data, episode, ax)


def gen(is_static, llm, info):
    # Set the style
    sns.set_theme(style="white", font="serif")

    fig = plt.figure(figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]))
    outer_grid = fig.add_gridspec(2, 1, hspace=0.4, wspace=0.25)

    axes = [
        {"ax": None, "ep": 0},
        {"ax": None, "ep": 0},
    ]

    y, x = 0, 0  # Start from the second column
    for env, args in info.items():
        print(f"Adding subplot ({y}, {x})")

        sns.set_theme(
            style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
        )
        ax = fig.add_subplot(outer_grid[y, x])
        axes[y]["ax"] = ax  # Adjust index to account for ghost axis

        if y == 0:
            episode = 100
        else:
            episode = 1600

        axes[y]["ep"] = episode

        d = gen_scenario_comparison_data(
            info,
            episode,
            llm,
            True,
        )
        ax1 = _gen_scenario_comparison_single_ep(None, d, episode, ax)

        ax1.get_legend().remove()

        y += 1
        if y >= 2:
            break

    axes[0]["ax"].sharex(axes[1]["ax"])
    axes[0]["ax"].tick_params(axis="x", which="both", labelbottom=False)
    axes[0]["ax"].set_xlabel("")

    axes[0]["ax"].set_title(f"Episode {axes[0]['ep']}", fontsize=CONSTS["FONT_SIZE"])
    axes[1]["ax"].set_title(f"Episode {axes[1]['ep']}", fontsize=CONSTS["FONT_SIZE"])

    axes[0]["ax"].set_ylabel("", fontsize=CONSTS["FONT_SIZE"])
    axes[1]["ax"].set_ylabel("", fontsize=CONSTS["FONT_SIZE"])

    fig.supylabel("Performance", fontsize=CONSTS["FONT_SIZE"], x=-0.01)

    handles, labels = axes[0]["ax"].get_legend_handles_labels()

    # fig.subplots_adjust(bottom=0.2)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.45),
        prop={"size": CONSTS["FONT_SIZE"]},
        # frameon=False,
    )

    plt.tight_layout()

    u.save_plot(
        fig,
        f"g_sim_scenario_comparison_{'static' if is_static else 'dynamic'}_all.pdf",
    )

    # plt.show()


if __name__ == "__main__":
    from master import info, llm_name

    gen(True, llm_name, info)
