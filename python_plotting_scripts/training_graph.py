import os

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from llm_graph import gen_llm, gen_llm_data
from matplotlib.lines import Line2D

import utils as u
from utils import CONSTS

BASE_PATH = os.path.join(u.get_root_dir(), "csvs_for_graphs/")


def gen_training_graph_data(
    name, hybrid_file, traditional_file, llm, break_point, is_static, ep_change_rate=None
):
    print(f"Plotting training performance graph for {'static' if is_static else 'dynamic'} {name}")

    if is_static:
        hybrid_file = f"s_{hybrid_file}_static_env.csv"
        traditional_file = f"s_{traditional_file}_static_env.csv"
    else:
        assert ep_change_rate is not None
        hybrid_file = f"d_{hybrid_file}_dynamic_env_{ep_change_rate}.csv"
        traditional_file = f"d_{traditional_file}_dynamic_env_{ep_change_rate}.csv"

    data_hybrid = pd.read_csv(
        os.path.join(BASE_PATH, "hybrid/", hybrid_file),
        header=0,
        names=[
            "episode",
            "hyb_alice_start",
            "hyb_alice_end",
            "hyb_alice_goal",
            "hyb_bob_start",
            "hyb_bob_end",
            "hyb_bob_goal",
            "hyb_performance",
            "hyb_scenario",
            "hyb_env_change_rate",
        ],
    )

    data_traditional = pd.read_csv(
        os.path.join(BASE_PATH, "marl/", traditional_file),
        header=0,
        names=[
            "episode",
            "tra_alice_start",
            "tra_alice_end",
            "tra_alice_goal",
            "tra_bob_start",
            "tra_bob_end",
            "tra_bob_goal",
            "tra_performance",
            "tra_scenario",
            "tra_env_change_rate",
        ],
    )

    # For data_hybrid_avg
    data_hybrid_avg = (
        data_hybrid.groupby("episode")
        .agg({"hyb_performance": ["median", "std", "count"]})
        .reset_index()
    )

    data_hybrid_avg.columns = ["episode", "hyb_performance_avg", "hyb_performance_sd", "hyb_n"]
    data_hybrid_avg["hyb_performance_se"] = data_hybrid_avg["hyb_performance_sd"] / np.sqrt(
        data_hybrid_avg["hyb_n"]
    )

    # For data_traditional_avg
    data_traditional_avg = (
        data_traditional.groupby("episode")
        .agg({"tra_performance": ["median", "std", "count"]})
        .reset_index()
    )

    data_traditional_avg.columns = [
        "episode",
        "tra_performance_avg",
        "tra_performance_sd",
        "tra_n",
    ]
    data_traditional_avg["tra_performance_se"] = data_traditional_avg[
        "tra_performance_sd"
    ] / np.sqrt(data_traditional_avg["tra_n"])

    # Merging the data
    data = pd.merge(data_hybrid_avg, data_traditional_avg, on="episode", how="left")

    print(f"hybrid_n: {data['hyb_n'].median()}")
    print(f"traditional_n: {data['tra_n'].median()}")
    print(data)

    return data


def _gen_training_graph_static(name, llm, data, break_point, is_static, parent_ax):
    sns.set_theme(
        style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
    )

    # Create two subplots for the broken axis
    if parent_ax is None:
        fig, parent_ax = plt.subplots(figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]))
    else:
        fig = parent_ax.figure

    # Create a sub-gridspec within the parent axis
    gs = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=parent_ax.get_subplotspec(),
        width_ratios=[0.5, 0.25, 0.25],
        wspace=0.1,
    )

    # Create two subplots for the broken axis
    ax1 = fig.add_subplot(gs[0, 0], sharey=parent_ax)
    ax2 = fig.add_subplot(gs[0, 1], sharey=parent_ax)
    ax3 = fig.add_subplot(gs[0, 2], sharey=parent_ax)

    # Function to plot data on a given axis
    def plot_on_axis(ax):
        sns.scatterplot(
            data=data,
            x="episode",
            y="tra_performance_avg",
            edgecolor=CONSTS["MARL_COLOUR"],
            label="MARL",
            s=CONSTS["POINT_SIZE"],
            color="none",
            ax=ax,
            marker=CONSTS["MARL_MARKER"],
            alpha=CONSTS["POINT_ALPHA"],
        )

        sns.scatterplot(
            data=data,
            x="episode",
            y="hyb_performance_avg",
            edgecolor=CONSTS["MARLIN_COLOUR"],
            label="MARLIN",
            s=CONSTS["POINT_SIZE"],
            color="none",
            ax=ax,
            marker=CONSTS["MARLIN_MARKER"],
            alpha=CONSTS["POINT_ALPHA"],
        )

    # Plot data on both axes
    plot_on_axis(ax1)
    plot_on_axis(ax2)

    d_llm = gen_llm_data(name, "llm_data/", f"LLM_{llm}_data.csv")
    gen_llm(name, "llm_data/", d_llm, ax3)

    # Set the limits for each subplot
    # ax1.set_xlim(0, break_point)
    # ax2.set_xlim(1400, 1600)

    actual_bp = break_point + 25

    ax1.set_xlim(0, actual_bp)
    ax1.set_xticks([0, break_point / 2, break_point])
    ax2.set_xlim(1375, 1600)
    ax2.set_xticks([1400, 1500, 1600])
    # ax3.set_xlim(ax3.get_xlim())  # Set appropriate limits for LLM data

    # Remove the right spine of the first subplot and the left spine of the second subplot
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Customize the plot
    ax1.set_xlabel("", fontsize=CONSTS["FONT_SIZE"])
    ax1.set_ylabel("", fontsize=CONSTS["FONT_SIZE"])
    ax2.set_xlabel("", fontsize=CONSTS["FONT_SIZE"])
    ax2.set_ylabel("", fontsize=CONSTS["FONT_SIZE"])
    ax3.set_ylabel("", fontsize=CONSTS["FONT_SIZE"])
    # fig.supxlabel("Episode", fontsize=CONSTS["FONT_SIZE"])
    # fig.supylabel("Performance", fontsize=CONSTS["FONT_SIZE"])

    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)  # Hide y-ticks on second subplot
    plt.setp(ax3.get_yticklabels(), visible=False)  # Hide y-ticks on third subplot
    plt.setp(ax1.get_xticklabels(), fontsize=CONSTS["FONT_SIZE"])
    plt.setp(ax2.get_xticklabels(), fontsize=CONSTS["FONT_SIZE"])
    plt.setp(ax3.get_xticklabels(), visible=False)

    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    parent_ax.set_xticks([])
    parent_ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    parent_ax.spines["top"].set_visible(False)
    parent_ax.spines["right"].set_visible(False)
    parent_ax.spines["bottom"].set_visible(False)
    parent_ax.spines["left"].set_visible(False)

    parent_ax.tick_params(axis="y", labelsize=CONSTS["FONT_SIZE"])

    # Remove top and right spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # ax1.spines["bottom"].set_visible(False)
    # ax1.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    # ax3.spines["bottom"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()

    # handles = ax1_handles + ax3_handles
    # labels = ax1_labels + ax3_labels

    # Remove legends from individual subplots
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    # fig.legend(
    #     handles,
    #     labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.015),
    #     fontsize=CONSTS["FONT_SIZE"],
    #     ncol=3,
    #     frameon=False,
    # )
    # # fig.subplots_adjust(bottom=0.2)

    # Add diagonal lines to indicate the break
    d = 0.04
    lw = 0.6  # Line width

    # Calculate the slope adjustment factor
    slope_factor = 5  # Because the left subplot is twice as wide as the right

    # Shift factor for moving the break indicators to the right on ax1
    shift = 0.02  # Adjust this value as needed

    # Get the axis color
    axis_color = ax1.spines["left"].get_edgecolor()

    # Left break signs (adjusted slope and shifted right)
    kwargs = dict(transform=ax1.transAxes, color=axis_color, clip_on=False, linewidth=lw)
    ax1.plot((1 - d + shift, 1 + d / slope_factor + shift), (-d, +d), **kwargs)
    # ax1.plot((1 - d + shift, 1 + d / slope_factor + shift), (1 - d, 1 + d), **kwargs)

    # Right break signs (original slope)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    plt.tight_layout()

    # If this is the top-level plot, add the legend
    if parent_ax is None:
        u.save_plot(
            fig,
            f"g_sim_training_perf_{'static' if is_static else 'dynamic'}_{name.replace(' ', '-')}.pdf",
        )
        # plt.show()

    return ax1, ax2, ax3


def _gen_static(is_static, llm, info):
    # Set the style
    sns.set_theme(style="white", font="serif")

    fig = plt.figure(figsize=(CONSTS["WIDTH"] * 3, CONSTS["HEIGHT"] * 2))
    outer_grid = fig.add_gridspec(2, 3, hspace=0.7, wspace=0.15)

    y, x = 0, 0
    for env, args in info.items():
        print(f"Adding subplot ({y}, {x})")

        sns.set_theme(style="white", font="serif")
        ax = fig.add_subplot(outer_grid[y, x])
        ax.set_title(env, fontsize=CONSTS["FONT_SIZE"])

        if x == 0:
            ax.set_ylabel("Performance", fontsize=CONSTS["FONT_SIZE"])

        if y == 1 or (y == 0 and x == 2):
            # plt.subplots_adjust(bottom=0.2)
            ax.set_xlabel("Episode", fontsize=CONSTS["FONT_SIZE"], labelpad=30)

        d = gen_training_graph_data(
            env,
            args["hybrid_file"],
            args["traditional_file"],
            llm,
            args["break_point"],
            is_static,
        )

        ax1, ax2, ax3 = _gen_training_graph_static(env, llm, d, args["break_point"], is_static, ax)

        if x != 0:
            ax1.set_ylabel("")
            ax1.set_yticklabels("")
            ax2.set_ylabel("")
            ax2.set_yticklabels("")

        if y != 1 and x != 2:
            ax1.set_xlabel("")
            # ax1.set_xticklabels("")
            ax2.set_xlabel("")
            # ax2.set_xticklabels("")

        x += 1
        if x >= 3:
            x = 0
            y += 1

        if y >= 2:
            break

    llm_handle = mpatches.Patch(color=CONSTS["LLM_COLOUR"], label="LLM")

    marlin_handle = Line2D(
        [0],
        [0],
        marker=CONSTS["MARLIN_MARKER"],
        markeredgecolor=CONSTS["MARLIN_COLOUR"],
        markerfacecolor="none",
        label="MARLIN",
        linestyle="None",
        alpha=CONSTS["POINT_ALPHA"],
    )

    marl_handle = Line2D(
        [0],
        [0],
        marker=CONSTS["MARL_MARKER"],
        markeredgecolor=CONSTS["MARL_COLOUR"],
        markerfacecolor="none",
        label="MARL",
        linestyle="None",
        alpha=CONSTS["POINT_ALPHA"],
    )

    fig.legend(
        handles=[llm_handle, marl_handle, marlin_handle],
        loc="center",
        # frameon=False,
        bbox_to_anchor=(0.75, 0.25),
        fontsize=CONSTS["FONT_SIZE"],
    )

    plt.tight_layout()

    u.save_plot(
        fig,
        f"g_sim_training_perf_{'static' if is_static else 'dynamic'}_all.pdf",
    )

    # plt.show()


def _gen_training_graph_dynamic(
    name, llm, data, break_point, is_static, parent_ax, ep_change_rate
):
    # sns.set_theme(
    #     style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
    # )

    # Create two subplots for the broken axis
    if parent_ax is None:
        fig, parent_ax = plt.subplots(figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]))
    else:
        fig = parent_ax.figure

    sns.scatterplot(
        data=data,
        x="episode",
        y="tra_performance_avg",
        edgecolor=CONSTS["MARL_COLOUR"],
        label="MARL",
        s=CONSTS["POINT_SIZE"],
        color="none",
        ax=parent_ax,
        marker=CONSTS["MARL_MARKER"],
        alpha=CONSTS["POINT_ALPHA"],
    )

    sns.scatterplot(
        data=data,
        x="episode",
        y="hyb_performance_avg",
        edgecolor=CONSTS["MARLIN_COLOUR"],
        label="MARLIN",
        s=CONSTS["POINT_SIZE"],
        color="none",
        ax=parent_ax,
        marker=CONSTS["MARLIN_MARKER"],
        alpha=CONSTS["POINT_ALPHA"],
    )

    parent_ax.axvline(
        x=ep_change_rate,
        color=CONSTS["EP_CHANGE_COLOUR"],
        linestyle="--",
        linewidth=1,
        label="Episode Change",
    )

    parent_ax.set_xlim(0, 1600)
    parent_ax.set_ylim(0.0, 1.0)

    parent_ax.set_xlabel("Episode", fontsize=CONSTS["FONT_SIZE"])
    parent_ax.set_ylabel("Performance", fontsize=CONSTS["FONT_SIZE"])

    plt.setp(parent_ax.get_xticklabels(), fontsize=CONSTS["FONT_SIZE"])
    plt.setp(parent_ax.get_yticklabels(), fontsize=CONSTS["FONT_SIZE"])

    parent_ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    parent_ax.spines["top"].set_visible(False)
    parent_ax.spines["right"].set_visible(False)

    parent_ax.tick_params(axis="y", labelsize=CONSTS["FONT_SIZE"])
    parent_ax.tick_params(axis="x", labelsize=CONSTS["FONT_SIZE"])

    plt.setp(parent_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Remove top and right spines
    parent_ax.spines["top"].set_visible(False)
    parent_ax.spines["right"].set_visible(False)

    # Remove legends from individual subplots
    parent_ax.get_legend().remove()

    plt.tight_layout()

    # If this is the top-level plot, add the legend
    if parent_ax is None:
        u.save_plot(
            fig,
            f"g_sim_training_perf_{'static' if is_static else 'dynamic'}_{name.replace(' ', '-')}.pdf",
        )
        # plt.show()

    return parent_ax


def _gen_dynamic(is_static, llm, info, ep_change_rate):
    # Set the style
    sns.set_theme(style="white", font="serif")

    fig = plt.figure(figsize=(CONSTS["WIDTH"] * 3, CONSTS["HEIGHT"] * 2))
    outer_grid = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.15)

    y, x = 0, 0

    for env, args in info.items():
        if env == "Maze Like Corridor":
            continue

        print(f"Adding subplot ({y}, {x})")

        # sns.set_theme(style="whitegrid", font="serif")
        sns.set_theme(
            style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
        )
        ax = fig.add_subplot(outer_grid[y, x])
        ax.set_title(env, fontsize=CONSTS["FONT_SIZE"])

        if x == 0:
            ax.set_ylabel("Performance", fontsize=CONSTS["FONT_SIZE"])

        if y == 1 or (y == 0 and x == 2):
            # plt.subplots_adjust(bottom=0.2)
            ax.set_xlabel("Episode", fontsize=CONSTS["FONT_SIZE"])

        d = gen_training_graph_data(
            env,
            args["hybrid_file"],
            args["traditional_file"],
            llm,
            args["break_point"],
            is_static,
            ep_change_rate,
        )

        ax1 = _gen_training_graph_dynamic(
            env, llm, d, args["break_point"], is_static, ax, ep_change_rate
        )

        if x != 0:
            ax1.set_ylabel("")
            ax1.set_yticklabels("")

        if y != 1:
            ax1.set_xlabel("")
            ax1.set_xticklabels("")

        x += 1
        if x >= 2:
            x = 0
            y += 1

        if y >= 2:
            break

    marlin_handle = Line2D(
        [0],
        [0],
        marker=CONSTS["MARLIN_MARKER"],
        markeredgecolor=CONSTS["MARLIN_COLOUR"],
        markerfacecolor="none",
        label="MARLIN",
        linestyle="None",
        alpha=CONSTS["POINT_ALPHA"],
    )

    marl_handle = Line2D(
        [0],
        [0],
        marker=CONSTS["MARL_MARKER"],
        markeredgecolor=CONSTS["MARL_COLOUR"],
        markerfacecolor="none",
        label="MARL",
        linestyle="None",
        alpha=CONSTS["POINT_ALPHA"],
    )

    ep_change_line = Line2D(
        [0], [0], color=CONSTS["EP_CHANGE_COLOUR"], linestyle="--", label="Episode Change"
    )

    fig.legend(
        handles=[marl_handle, marlin_handle, ep_change_line],
        loc="lower center",
        # frameon=False,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=CONSTS["FONT_SIZE"],
        # nrow=1,
        ncol=4,
    )

    plt.tight_layout()

    u.save_plot(
        fig,
        f"g_sim_training_perf_{'static' if is_static else 'dynamic'}_all.pdf",
    )

    # plt.show()


def gen(is_static, llm, info, ep_change_rate=None):
    if is_static:
        return _gen_static(is_static, llm, info)
    else:
        assert ep_change_rate is not None
        return _gen_dynamic(is_static, llm, info, ep_change_rate)


if __name__ == "__main__":
    llm_name = "Meta-Llama-3.1-8B-Instruct"
    env_change_rate = 1000

    asymmetrical_two_hybrid_file = "Asymmetrical_Two_Slot_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
    asymmetrical_two_traditional_file = (
        "Asymmetrical_Two_Slot_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"
    )

    single_slot_hybrid_file = "Single_Slot_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
    single_slot_traditional_file = (
        "Single_Slot_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"
    )

    symmetrical_two_hybrid_file = "Symmetrical_Two_Slot_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
    symmetrical_two_traditional_file = (
        "Symmetrical_Two_Slot_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"
    )

    two_path_hybrid_file = "Two_Path_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
    two_path_traditional_file = "Two_Path_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"

    maze_hybrid_file = "Maze_Like_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
    maze_traditional_file = "Maze_Like_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"

    info = {
        "Asymmetrical Two Slot Corridor": {
            "hybrid_file": asymmetrical_two_hybrid_file,
            "traditional_file": asymmetrical_two_traditional_file,
            "break_point": 200,
        },
        "Single Slot Corridor": {
            "hybrid_file": single_slot_hybrid_file,
            "traditional_file": single_slot_traditional_file,
            "break_point": 200,
        },
        "Symmetrical Two Slot Corridor": {
            "hybrid_file": symmetrical_two_hybrid_file,
            "traditional_file": symmetrical_two_traditional_file,
            "break_point": 200,
        },
        "Two Path Corridor": {
            "hybrid_file": two_path_hybrid_file,
            "traditional_file": two_path_traditional_file,
            "break_point": 200,
        },
        "Maze Like Corridor": {
            "hybrid_file": maze_hybrid_file,
            "traditional_file": maze_traditional_file,
            "break_point": 500,
        },
    }

    gen(True, llm_name, info)
