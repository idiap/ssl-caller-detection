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

from sklearn.manifold import TSNE

import utils


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Convert the extracted features from pre-trained models into fixed-length functionals using mean and std averaging."
    )
    parser.add_argument(
        "-n",
        "--expname",
        help="""
        Name of the folder name in the experiment directory `expdir` where the embeddings
        extracted from the `extract_features.py` files are stored.
        Must be an existing folder. Eg: 'modified_cpc'")
        """,
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
    parser.add_argument(
        "-g",
        "--groups",
        default=100,
        help="""
        Number of groups to split the train data into to make Gaussians.")
        """,
    )
    return parser


def heatmap():
    # Make a plot with imshow()
    # Do it using inter_distances.items() instead of dist_names
    fig, axes = plt.subplots(3, 1, figsize=(10, 20))
    for i, (k, v) in enumerate(inter_distances.items()):
        # Make a 10x10 matrix
        matrix = np.zeros((10, 10))
        for c1 in callers:
            for c2 in callers:
                if c1 != c2:
                    matrix[c1 - 1, c2 - 1] = inter_distances[k][c1][c2]

        # Plot the matrix
        sns.heatmap(matrix, ax=axes[i])
        axes[i].set_title(k)
        axes[i].set_xlabel("Caller ID")
        axes[i].set_ylabel("Caller ID")

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_path, f"inter_distances_heatmap.png"), dpi=300
    )
    plt.close(fig)


def inter_distances_plot():
    # Plot the same inter distances in a 2x5 grid
    for k, v in inter_distances.items():
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        c = 1
        for i in range(len(axes)):
            for j in range(len(axes[i])):
                sns.distplot(inter_distances[k][c], bins=100, ax=axes[i, j])
                axes[i, j].set_title(f"Caller {c}")
                c += 1

        # Fix x-axis limits
        # if k == "inter_group_kl_distances":
        #     for ii in range(len(axes)):
        #         for jj in range(len(axes[i])):
        #             axes[ii, jj].set_xlim(0, 0.25)
        # elif k == "inter_group_bhatt_dist":
        #     for ii in range(len(axes)):
        #         for jj in range(len(axes[i])):
        #             axes[ii, jj].set_xlim(-9.2, -8.6)
        # elif k == "inter_group_bhatt_coeff":
        #     for ii in range(len(axes)):
        #         for jj in range(len(axes[i])):
        #             axes[ii, jj].set_xlim(6000, 9000)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"{k}_distances.png"), dpi=300)
        plt.close(fig)


def another_inter_plot():
    # Plot the distribution of inter distances
    for k, v in inter_distances.items():
        fig, ax = plt.subplots()
        for c in callers:
            sns.distplot(inter_distances[k][c], bins=100, label=f"Caller {c}")
        ax.set_title(k)
        ax.legend()
        fig.savefig(os.path.join(save_path, f"{k}_dist.png"), dpi=300)
        plt.close(fig)


def intra_distances_plot():
    # Plot the distribution of intra distances
    for k, v in intra_distances.items():
        fig, ax = plt.subplots()
        for c in callers:
            sns.distplot(intra_distances[k][c], bins=100, label=f"Caller {c}")
        ax.set_title(k)
        ax.legend()
        fig.savefig(os.path.join(save_path, f"{k}_intra_dist.png"), dpi=300)
        plt.close(fig)


def plot_heatmaps(inter_data, intra_data, save_path, exp_name, app=""):
    # Process inter-group data, i.e. across callers
    inter_dist = []
    for c, dist in inter_data["kl_dist"].items():
        inter_dist.append((c[0], c[1], "inter", "kl", sum(dist) / len(dist)))
    for c, dist in inter_data["bhatt_dist"].items():
        inter_dist.append((c[0], c[1], "inter", "bhatt", sum(dist) / len(dist)))

    # Process intra-group data, i.e. within callers
    intra_dist = []
    for c, dist in intra_data["kl_dist"].items():
        intra_dist.append((c, c, "intra", "kl", sum(dist) / len(dist)))
    for c, dist in intra_data["bhatt_dist"].items():
        intra_dist.append((c, c, "intra", "bhatt", sum(dist) / len(dist)))

    # Create dataframe
    dist_df = pd.DataFrame(
        inter_dist + intra_dist,
        columns=[
            "caller_1",
            "caller_2",
            "type",
            "metric",
            "distance",
        ],
    )

    # Create dataframes for inter- and intra-speaker distances
    inter_df = pd.DataFrame(
        inter_dist,
        columns=["caller_1", "caller_2", "type", "metric", "distance"],
    )
    intra_df = pd.DataFrame(
        intra_dist,
        columns=["caller_1", "caller_2", "type", "metric", "distance"],
    )

    # Pivot dataframes to create heatmaps
    inter_pivot = inter_df.pivot_table(
        index="caller_1", columns="caller_2", values="distance", aggfunc="mean"
    )
    intra_pivot = intra_df.pivot_table(
        index="caller_1", columns="caller_2", values="distance", aggfunc="mean"
    )

    # Create dataframes for each distance
    kl_inter_df = inter_df[inter_df["metric"] == "kl"]
    bhatt_inter_df = inter_df[inter_df["metric"] == "bhatt"]
    kl_intra_df = intra_df[intra_df["metric"] == "kl"]
    bhatt_intra_df = intra_df[intra_df["metric"] == "bhatt"]

    # Plot heatmaps for each distance without aggregation
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    sns.heatmap(
        kl_inter_df.pivot_table(
            index="caller_1", columns="caller_2", values="distance"
        ),
        cmap="YlGnBu",
        ax=ax1,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
    )
    ax1.set_title("Inter-Caller KL Distances")
    sns.heatmap(
        bhatt_inter_df.pivot_table(
            index="caller_1", columns="caller_2", values="distance"
        ),
        cmap="YlGnBu",
        ax=ax2,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
    )
    ax2.set_title("Inter-Caller Bhattacharyya Distances")
    plt.savefig(
        os.path.join(
            save_path, f"heatmaps/inter/{exp_name}_inter_distances{app}.png"
        )
    )
    plt.close()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    sns.heatmap(
        kl_intra_df.pivot_table(
            index="caller_1", columns="caller_2", values="distance"
        ),
        cmap="YlGnBu",
        ax=ax1,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
    )
    ax1.set_title("Intra-Caller KL Distances")
    sns.heatmap(
        bhatt_intra_df.pivot_table(
            index="caller_1", columns="caller_2", values="distance"
        ),
        cmap="YlGnBu",
        ax=ax2,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
    )
    ax2.set_title("Intra-Caller Bhattacharyya Distances")
    plt.savefig(
        os.path.join(
            save_path, f"heatmaps/intra/{exp_name}_intra_distances{app}.png"
        )
    )
    plt.close()

    # Plot average heatmaps with aggregation across KL and BC distances
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    sns.heatmap(
        inter_pivot,
        cmap="YlGnBu",
        ax=ax1,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
    )
    ax1.set_title("Inter-Caller Distances")
    sns.heatmap(
        intra_pivot,
        cmap="YlGnBu",
        ax=ax2,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
    )
    ax2.set_title("Intra-Caller Distances")
    plt.savefig(
        os.path.join(save_path, f"heatmaps/{exp_name}_distances{app}.png")
    )
    plt.close()


def plot_speaker_intra_distances(intra_data, save_path, exp_name, app=""):
    # Plot KDE for each speaker
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    # KL
    for c, dist in intra_data["kl_dist"].items():
        sns.kdeplot(
            dist, ax=ax1, label=f"Caller {c}", fill=True, log_scale=False
        )
    ax1.set_title(f"KL Distances")
    ax1.legend()
    # Bhattacharyya
    for c, dist in intra_data["bhatt_dist"].items():
        sns.kdeplot(
            dist, ax=ax2, label=f"Caller {c}", fill=True, log_scale=False
        )
    ax2.set_title(f"Bhattacharyya Distances")
    ax2.legend()
    plt.savefig(
        os.path.join(
            save_path, f"kde/intra/{exp_name}_kde_intra_distances{app}.png"
        )
    )
    plt.close()

    # Plot KDE for each speaker
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    # KL
    for c, dist in intra_data["kl_dist"].items():
        sns.kdeplot(
            dist, ax=ax1, label=f"Caller {c}", fill=True, log_scale=True
        )
    ax1.set_title(f"Log KL Distances")
    ax1.legend()
    # Bhattacharyya
    for c, dist in intra_data["bhatt_dist"].items():
        sns.kdeplot(
            dist, ax=ax2, label=f"Caller {c}", fill=True, log_scale=True
        )
    ax2.set_title(f"Log Bhattacharyya Distances")
    ax2.legend()
    plt.savefig(
        os.path.join(
            save_path, f"kde/intra/{exp_name}_kde_intra_log_distances{app}.png"
        )
    )
    plt.close()


def plot_speaker_inter_distances(inter_data, save_path, exp_name, app=""):
    # Plot KDE across speakers
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    # KL
    for c, dist in inter_data["kl_dist"].items():
        sns.kdeplot(
            dist, ax=ax1, label=f"Caller {c}", fill=True, log_scale=False
        )
    ax1.set_title(f"KL Distances")
    ax1.legend()
    # Bhattacharyya
    for c, dist in inter_data["bhatt_dist"].items():
        sns.kdeplot(
            dist, ax=ax2, label=f"Caller {c}", fill=True, log_scale=False
        )
    ax2.set_title(f"Bhattacharyya Distances")
    ax2.legend()
    plt.savefig(
        os.path.join(
            save_path, f"kde/inter/{exp_name}_kde_inter_distances{app}.png"
        )
    )
    plt.close()

    # Plot KDE across speakers
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    # KL
    for c, dist in inter_data["kl_dist"].items():
        sns.kdeplot(
            dist, ax=ax1, label=f"Caller {c}", fill=True, log_scale=True
        )
    ax1.set_title(f"Log KL Distances")
    ax1.legend()
    # Bhattacharyya
    for c, dist in inter_data["bhatt_dist"].items():
        sns.kdeplot(
            dist, ax=ax2, label=f"Caller {c}", fill=True, log_scale=True
        )
    ax2.set_title(f"Log Bhattacharyya Distances")
    ax2.legend()
    plt.savefig(
        os.path.join(
            save_path, f"kde/inter/{exp_name}_kde_inter_log_distances{app}.png"
        )
    )
    plt.close()


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    # Save path
    for plot_type in ["kde", "heatmaps"]:
        for group_type in ["inter", "intra"]:
            save_path = os.path.join(
                args.savedir, args.expname, args.task, plot_type, group_type
            )
            os.makedirs(save_path, exist_ok=True)
    callers = np.arange(1, 11)  # 10 callers -> 1 to 10
    app = "_sub"
    save_path = os.path.join(args.savedir, args.expname, args.task)

    # Read distances from the pickle file
    # Inter -> distances across callers
    # Intra -> distances within callers
    inter_distances_path_sub = os.path.join(
        args.expdir,
        args.expname + "_distris",
        args.task,
        args.expname + f"_distris_inter_distances{app}.pkl",
    )

    intra_distances_path_sub = os.path.join(
        args.expdir,
        args.expname + "_distris",
        args.task,
        args.expname + f"_distris_intra_distances{app}.pkl",
    )

    inter_distances_path = os.path.join(
        args.expdir,
        args.expname + "_distris",
        args.task,
        args.expname + f"_distris_inter_distances.pkl",
    )

    intra_distances_path = os.path.join(
        args.expdir,
        args.expname + "_distris",
        args.task,
        args.expname + f"_distris_intra_distances.pkl",
    )

    print("Loading distances ...")
    inter_distances = utils.load_features(inter_distances_path)
    inter_distances_sub = utils.load_features(inter_distances_path_sub)

    intra_distances = utils.load_features(intra_distances_path)
    intra_distances_sub = utils.load_features(intra_distances_path_sub)

    print("Plotting ...")
    plot_heatmaps(
        inter_distances, intra_distances, save_path, args.expname, app=""
    )
    plot_heatmaps(
        inter_distances_sub,
        intra_distances_sub,
        save_path,
        args.expname,
        app=app,
    )

    plot_speaker_intra_distances(
        intra_distances, save_path, args.expname, app=""
    )
    plot_speaker_intra_distances(
        intra_distances_sub, save_path, args.expname, app=app
    )

    plot_speaker_inter_distances(
        inter_distances, save_path, args.expname, app=""
    )
    plot_speaker_inter_distances(
        inter_distances_sub, save_path, args.expname, app=app
    )
