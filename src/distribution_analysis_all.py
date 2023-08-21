# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

# Make a TSNE plot showing the different caller functionals
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from sklearn.manifold import TSNE

import utils


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Convert the extracted features from pre-trained models into fixed-length functionals using mean and std averaging."
    )
    parser.add_argument(
        "-p",
        "--expdir",
        help="""
        Path of the experiment directory where the embeddings
        extracted from the `extract_features.py` folders are stored.")
        """,
    )
    parser.add_argument(
        "-o",
        "--savedir",
        help="""
        Path where to save the functional tensors.")
        """,
    )
    parser.add_argument(
        "-t",
        "--task",
        help="""
        Call-type classification (`calltypeID`) or caller recognition (`marmosetID`).")
        """,
        required=True,
    )
    return parser


def plot_model_distances(
    models, app, save_path, dist_type, group_type, m_names
):
    # Make subplot of 2x5, one plot for each model and plot the KDE of the KL distances
    m = 0
    max_value = 0
    fig, axs = plt.subplots(2, 5, figsize=(15, 8))
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            # Read distances
            distances_path = os.path.join(
                args.expdir,
                models[m] + "_distris",
                args.task,
                models[m] + f"_distris_{group_type}_distances{app}.pkl",
            )

            data = utils.load_features(distances_path)

            # Plot KDE for each speaker
            for c, dist in data[dist_type].items():
                sns.kdeplot(
                    dist,
                    ax=axs[i, j],
                    label=f"Caller {c}",
                    fill=True,
                    log_scale=False,
                )
                max_value = max(max_value, max(dist))

            # Config
            axs[i, j].set_title(f"{m_names[models[m]]}")

            # Set x limit to half of the max value
            axs[i, j].set_xlim(0, max_value / 2)
            axs[i, j].set_ylabel("")

            # Increment model counter
            m += 1

            # reset max_value
            max_value = 0

    # Write y labels for [0,0] and [1,0]. X labels for [1,0] till [1,5]. Use a loop
    x_name = (
        "KL-divergence" if dist_type == "kl_dist" else "Bhattacharyya distance"
    )

    axs[0, 0].set_ylabel("Density")
    axs[1, 0].set_ylabel("Density")
    # axs[1, 0].set_xlabel(x_name)
    # axs[1, 1].set_xlabel(x_name)
    # axs[1, 2].set_xlabel(x_name)
    # axs[1, 3].set_xlabel(x_name)
    # axs[1, 4].set_xlabel(x_name)

    # Show legend at the bottom
    if group_type == "intra":
        plt.legend(
            *axs[0, 0].get_legend_handles_labels(),
            loc="lower center",
            ncol=10,
            bbox_to_anchor=(0.5, 0.0),
            bbox_transform=fig.transFigure,
        )

    # fig.suptitle(f"Inter-group {x_name}", fontsize=16)
    plt.savefig(
        os.path.join(
            save_path,
            f"kde_{group_type}_distances_{dist_type}_all_models{app}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def get_paper_names():
    model_names = {
        "apc_360hr": "APC",
        "data2vec_base_960": "Data2Vec",
        "distilhubert_base": "DistilHubert",
        "hubert_base": "Hubert",
        "mockingjay_logMelBase_T_AdamW_b32_200k_100hr": "Mockingjay",
        "modified_cpc": "Mod-CPC",
        "npc_360hr": "NPC",
        "tera_logMelBase_T_F_M_AdamW_b32_200k_100hr": "TERA",
        "vq_apc_360hr": "VQ-APC",
        "wav2vec2_base_960": "Wav2Vec2",
        "wavlm_base": "WavLM",
    }

    classifier_names = {
        "LinearSVC": "L_SVM",
        "SVC": "SVM",
        "RF": "RF",
        "AB": "AB",
    }

    return model_names, classifier_names


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    # Variables
    callers = np.arange(1, 11)  # 10 callers -> 1 to 10
    app = ""  # appendix (sub or non-sub)
    save_path = os.path.join(args.savedir, args.task)
    os.makedirs(save_path, exist_ok=True)

    m_names, c_names = get_paper_names()

    # Read distances for all models

    # Read yaml file `models.yaml`
    with open("models.yaml", "r") as f:
        models = yaml.load(f, Loader=yaml.FullLoader)["model"]

    # Plot
    for dist_type in ["kl_dist", "bhatt_dist"]:
        for group_type in ["inter", "intra"]:
            for app in ["", "_sub"]:
                plot_model_distances(
                    models, app, save_path, dist_type, group_type, m_names
                )

    # plot_model_distances(models, app, save_path, "kl_dist", "inter")
    # plot_model_distances(models, app, save_path, "bhatt_dist", "inter")

    # plot_model_distances(models, app, save_path, "kl_dist", "intra")
    # plot_model_distances(models, app, save_path, "bhatt_dist", "intra")
