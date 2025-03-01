import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

#######################
# Constants
#######################
CONSTS = {
    "FONT_SIZE": 8,
    "WIDTH": 4,
    "HEIGHT": 1.5,
    "UNITS": "in",
    "DPI": 300,
    "MARLIN_COLOUR": "#4c72b0",  # "#A708F7",
    "MARL_COLOUR": "#dd8452",  # "#F7A708",
    "LLM_COLOUR": "#55a868",  # "#08F7A7",
    "EP_CHANGE_COLOUR": "#e30cc9",
    "MARLIN_MARKER": "^",
    "MARL_MARKER": "D",
    "POINT_ALPHA": 1,
    "POINT_SIZE": 5,
    "LINE_WIDTH": 0.5,
}


# Set up the plot style
plt.rcParams.update(
    {
        "font.size": CONSTS["FONT_SIZE"],
        # "font.family": "sans-serif",
        # "font.sans-serif": "Helvetica",
        "figure.figsize": (CONSTS["WIDTH"], CONSTS["HEIGHT"]),
        "figure.dpi": CONSTS["DPI"],
        "lines.markersize": CONSTS["POINT_SIZE"] * 2,  # matplotlib uses diameter, not radius
        "lines.linewidth": CONSTS["LINE_WIDTH"],
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

# To use these settings in a plot:
# fig, ax = plt.subplots()
# ax.plot(x, y, color=MARLIN_COLOUR, alpha=POINT_ALPHA)
# plt.savefig('plot.png', dpi=DPI, bbox_inches='tight')


#######################
# Helper functions
#######################
def capitalise_words(string):
    words = string.replace("-", " ").split()

    if words[0].lower() == "gpt":
        words[0] = words[0].upper()
    else:
        words[0] = words[0].capitalize()

    words[1:] = [word.capitalize() for word in words[1:]]

    return " ".join(words)


def join_csvs(path, files):
    filenames = [os.path.join(path, file) for file in files]
    return pd.concat([pd.read_csv(filename) for filename in filenames], ignore_index=True)


def combine_data_frames(frames):
    return pd.concat(frames, ignore_index=True)


def get_root_dir():
    return "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])


def save_plot(fig, filename):
    width = fig.get_figwidth()
    height = fig.get_figheight()
    output_dir = os.path.join(get_root_dir(), "python_plotting_scripts/", "graphs/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)

    with PdfPages(filepath) as pdf:
        fig.set_size_inches(width, height)
        fig.savefig(pdf, format="pdf", dpi=CONSTS["DPI"], bbox_inches="tight")


def remove_pdf_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
