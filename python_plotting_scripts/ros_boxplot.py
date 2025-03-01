import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from llm_graph import gen_llm, gen_llm_data

import utils as u
from utils import CONSTS

BASE_PATH = os.path.join(u.get_root_dir(), "ros_results/")


def gen_ros_boxplot_data(name, hybrid_file, traditional_file, llm, is_static):
    print(f"Plotting performance graph for {'static' if is_static else 'dynamic'} {name}")

    data_hybrid = pd.read_csv(
        os.path.join(BASE_PATH, hybrid_file),
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
        os.path.join(BASE_PATH, traditional_file),
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

    # Combine the dataframes
    data = pd.merge(data_hybrid, data_traditional, on="episode", how="left")

    # Filter the data
    data = data[data["episode"].isin([100, 350, 600, 850, 1100, 1350, 1600])]

    # Print the entire DataFrame
    print(data)

    # Calculate and print the median of hyb_n and tra_n
    # print(f"hybrid_n: {data['hyb_n'].median()}")
    # print(f"traditional_n: {data['tra_n'].median()}")

    # Reshape the data from wide to long format
    data_long = pd.melt(
        data,
        id_vars=[col for col in data.columns if col not in ["tra_performance", "hyb_performance"]],
        value_vars=["tra_performance", "hyb_performance"],
        var_name="model",
        value_name="performance",
    )

    # Rename the model values
    data_long["model"] = data_long["model"].map(
        {"tra_performance": "MARL", "hyb_performance": "MARLIN"}
    )

    # print(data_long)
    return data_long


def gen_ros_boxplot(name, llm, data, ax=None):
    shared_ax = False if ax is None else True

    # Set the style
    sns.set_theme(
        style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
    )

    # Create the plot
    if not shared_ax:
        fig, ax = plt.subplots(figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]))

    # Create the boxplot
    sns.boxplot(
        x="episode",
        y="performance",
        hue="model",
        data=data,
        palette={"MARLIN": CONSTS["MARLIN_COLOUR"], "MARL": CONSTS["MARL_COLOUR"]},
        width=0.75,
        linewidth=0.5,
        fliersize=3,
        ax=ax,
        saturation=1,
    )

    # Customize the plot
    ax.set_xlabel("Episode", fontsize=CONSTS["FONT_SIZE"])
    ax.set_ylabel("Performance", fontsize=CONSTS["FONT_SIZE"])
    plt.yticks([0, 0.25, 0.50, 0.75, 1.0], fontsize=CONSTS["FONT_SIZE"])
    # plt.xticks(range(0, 1800, 200), fontsize=CONSTS["FONT_SIZE"])

    # Customize the legend
    # if not shared_ax:
    # fig.subplots_adjust(bottom=0.2)
    ax.legend(
        loc="upper center",  # Center the legend horizontally
        bbox_to_anchor=(0.5, -0.65),  # Move the legend below the x-axis
        fontsize=CONSTS["FONT_SIZE"],
        ncol=2,  # Keeps the legend compact
        # frameon=False,  # Removes legend background for a cleaner look
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout and display
    plt.tight_layout()

    if not shared_ax:
        u.save_plot(fig, f"g_real_{name.replace(' ', '-')}.pdf")
        # plt.show()


def gen(is_static, env, llm):
    # Set the style
    sns.set_theme(
        style="whitegrid", font="serif", rc={"axes.edgecolor": "black", "axes.linewidth": 1}
    )

    # Create a figure with a 1x2 grid
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(CONSTS["WIDTH"], CONSTS["HEIGHT"]),
        gridspec_kw={"width_ratios": [1, 0.3]},
        sharey=True,
    )

    d_models = gen_ros_boxplot_data(
        env,
        "hybrid_ros_results.csv",
        "marl_ros_results.csv",
        llm,
        True,
    )

    d_llm = gen_llm_data(env, "ros_results/", "llm_ros_results.csv")

    gen_ros_boxplot(env, llm, d_models, ax1)
    gen_llm(env, "ros_results/", d_llm, ax2)

    # ax1.set_xticks(range(1600))

    ax1.tick_params(axis="both", which="major", labelsize=CONSTS["FONT_SIZE"])
    ax2.tick_params(axis="both", which="minor", labelsize=CONSTS["FONT_SIZE"])

    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Remove any existing legends
    ax1.get_legend().remove() if ax1.get_legend() else None
    ax2.get_legend().remove() if ax2.get_legend() else None

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)

    # Collect handles and labels from both subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine handles and labels
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Add a common legend at the bottom
    # fig.subplots_adjust(bottom=0.2)
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.15),
        prop={"size": CONSTS["FONT_SIZE"]},
        # frameon=False,
    )

    u.save_plot(fig, f"g_real_{env.replace(' ', '-')}_{llm}.pdf")

    # Display the plot
    # plt.show()


# d = gen_ros_boxplot_data(
#     "Maze Like Corridor",
#     "hybrid_ros_results.csv",
#     "marl_ros_results.csv",
#     "Meta-Llama-3.1-8B-Instruct",
#     True,
# )
# gen_ros_boxplot("Maze Like Corridor", "Meta-Llama-3.1-8B-Instruct", d)
