import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils as u
from utils import CONSTS

BASE_PATH = u.get_root_dir()


def gen_llm_data(env, dir, file):
    print(
        f"Processing data for {'all environments' if env is None else env} with {os.path.join(dir, file)}"
    )

    # Read the CSV file
    llm_data = pd.read_csv(
        os.path.join(BASE_PATH, dir, file),
        header=0,
        names=[
            "llm",
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

    print(f"llm data {llm_data}")

    # If env is specified, filter the data, otherwise process all environments
    if env is not None:
        env = env.replace(" ", "_")
        llm_data = llm_data[llm_data["scenario"] == env]

    llm_data_avg = (
        llm_data.groupby("scenario").agg({"performance": ["median", "std", "count"]}).reset_index()
    )

    # Flatten the column names
    llm_data_avg.columns = ["scenario", "performance_avg", "performance_sd", "n"]

    # Calculate the standard error
    llm_data_avg["performance_se"] = llm_data_avg["performance_sd"] / np.sqrt(llm_data_avg["n"])

    # Print data and statistics
    print(llm_data)
    print(llm_data_avg)
    print(llm_data["performance"].min())
    print(llm_data["performance"].max())
    print(f"Overall median: {llm_data_avg['performance_avg'].median()}")

    # Convert scenario to categorical and set the order
    llm_data["scenario"] = pd.Categorical(
        llm_data["scenario"], categories=llm_data["scenario"].unique(), ordered=True
    )

    return llm_data


def _gen_llm_single_env(env, dir, data, ax=None):
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
        data=data,
        color=CONSTS["LLM_COLOUR"],
        label="LLM",
        width=0.75,
        linewidth=0.5,
        fliersize=3,
        saturation=1.0,
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
            f"g_llm_{'real' if 'ros' in dir else 'sim'}_{env.replace(' ', '-') if env is not None else 'all_envs'}.pdf",
        )
        # plt.show()


def _gen_llm_multiple_env(env, dir, data, ax=None):
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
        hue="scenario",
        data=data,
        width=0.75,
        linewidth=0.5,
        fliersize=3,
        saturation=1,
    )

    # Customize the plot
    plt.ylim(0, 1)
    plt.xticks(
        range(len(data["scenario"].unique())),
        [
            s.replace("_", "\n")  # .replace(" Corridor", "\nCorridor")
            for s in data["scenario"].unique()
        ],
        rotation=0,
    )

    # Legend
    # if not shared_ax:
    #     # fig.subplots_adjust(bottom=0.2)

    # plt.legend(
    #     loc="upper right",
    #     # frameon=False,
    #     fontsize=CONSTS["FONT_SIZE"],
    # )

    # Remove axes and labels
    plt.ylabel("Performance", fontsize=CONSTS["FONT_SIZE"])
    plt.xlabel("", fontsize=CONSTS["FONT_SIZE"])
    plt.yticks([0, 0.25, 0.50, 0.75, 1.0], fontsize=CONSTS["FONT_SIZE"])
    plt.xticks(fontsize=CONSTS["FONT_SIZE"])

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
            f"g_llm_{'real' if 'ros' in dir else 'sim'}_{env.replace(' ', '-') if env is not None else 'all_envs'}.pdf",
        )
        # plt.show()


def gen_llm(env, dir, data, ax=None):
    if env is not None:
        print("Plotting one env")
        _gen_llm_single_env(env, dir, data, ax)
    else:
        print("Plotting multiple envs")
        _gen_llm_multiple_env(env, dir, data, ax)


def gen(llm):
    d = gen_llm_data(None, "llm_data/", f"LLM_{llm}_data.csv")
    gen_llm(None, "llm_data/", d)


if __name__ == "__main__":
    d = gen_llm_data("Maze_Like_Corridor", "ros_results/", "llm_ros_results.csv")
    gen_llm("Maze_Like_Corridor", "ros_results/", d)

    d = gen_llm_data(None, "llm_data/", "LLM_Meta-Llama-3.1-8B-Instruct_data.csv")
    gen_llm(None, "llm_data/", d)
