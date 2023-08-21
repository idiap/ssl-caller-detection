# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>


import argparse
import os

import numpy as np

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import utils

from gaussian import Gaussian


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
    parser.add_argument(
        "-s",
        "--standardize",
        default=False,
        help="""
        Standardize the embeddings before computing the Gaussians.")
        """,
    )
    parser.add_argument(
        "-d",
        "--downsample",
        action=argparse.BooleanOptionalAction,
        help=""",
        Downsample the data to have equal samples per class.")
        """,
    )
    return parser


def standardise_data(data):
    """
    Standardise the data using sklearn.preprocessing.StandardScaler.
    """
    print("Standardizing embeddings...")

    # v-stack the X data and use sklearn.preprocessing.StandardScaler
    # to fit_transform the data, then un-stack it again
    X_train = np.vstack(data)

    # Fit and transform the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Split X_train back into a np.array of np.arrays with their original shapes
    X_train = np.split(X_train, np.cumsum([len(x) for x in data])[:-1])

    return X_train


def compute_intra_group_distances(gaussians_dict, callers, groups):
    """
    Compute the intra-caller-group distances between the Gaussians.
    Returns the KL divergence and Bhattacharyya distance.
    """
    kl_dist = {}
    bhatt_dist = {}

    for c in tqdm(callers):
        kl_dist[c] = []
        bhatt_dist[c] = []

        # Iterate over groups within each caller with a double index
        for i in range(groups):
            for j in range(groups):
                # We don't want distances between the same group
                if i != j:
                    # Get the two Gaussians
                    g1 = gaussians_dict[c][i]
                    g2 = gaussians_dict[c][j]

                    # Calculate the KL distance
                    kl_dist[c].append(Gaussian.D_KL_log(g1, g2).item())

                    # Calculate the Bhattacharyya distance
                    bhatt_dist[c].append(Gaussian.D_Bhatt_log(g1, g2).item())

        # Convert to np array
        kl_dist[c] = np.array(kl_dist[c])
        bhatt_dist[c] = np.array(bhatt_dist[c])

    return kl_dist, bhatt_dist


def compute_inter_group_distances(gaussians_dict, callers, groups):
    """
    Compute the inter-caller-group distances between the Gaussians.
    Returns the KL divergence and Bhattacharyya distance.
    """
    kl_dist = {}
    bhatt_dist = {}

    # Iterate over callers with double indices
    for c_i in tqdm(callers):
        for c_j in callers:
            # We don't want distances between the same caller
            if c_i != c_j:
                pair = (c_i, c_j)
                kl_dist[pair] = []
                bhatt_dist[pair] = []

                # Iterate over groups within the two chosen callers with a double index
                for i in range(groups):
                    for j in range(groups):
                        # We don't want distances between the same group
                        if i != j:
                            # Get the two Gaussians
                            g1 = gaussians_dict[c_i][i]
                            g2 = gaussians_dict[c_j][j]

                            # Calculate the KL distance
                            kl_dist[pair].append(
                                Gaussian.D_KL_log(g1, g2).item()
                            )

                            # Calculate the Bhattacharyya distance
                            bhatt_dist[pair].append(
                                Gaussian.D_Bhatt_log(g1, g2).item()
                            )

                # Convert to np array
                kl_dist[pair] = np.array(kl_dist[pair])
                bhatt_dist[pair] = np.array(bhatt_dist[pair])

    return kl_dist, bhatt_dist


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    # Task
    task = args.task
    num_groups = args.groups

    # Save path
    save_path = os.path.join(args.savedir, args.expname, task)
    os.makedirs(save_path, exist_ok=True)

    # Load the mapping between uids and pickleids
    uids_to_pickleid = utils.load_features(
        "marmoset_lists/uids_to_pickleid.pkl"
    )
    pickleid_to_uids = utils.load_features(
        "marmoset_lists/pickleid_to_uids.pkl"
    )

    # Read embeddings
    print("Reading embeddings...")
    sets = ["train"]
    X, y = utils.load_embeddings(
        args.expdir,
        args.expname.replace("distris", "feats"),
        task,
        uids_to_pickleid,
        sets,
    )

    # Downsample the data
    if args.downsample:
        X_train, y_train = utils.downsample_data(X, y, num_samples=2681)

    # Standardize the data if needed
    X_train = standardise_data(X_train) if args.standardize else X_train
    y_train = y_train if args.downsample else y["y_train"]

    print(
        f"X_train.shape: {X_train.shape}",
        f"y_train.shape: {y_train.shape}",
    )

    callers = sorted(np.unique(y_train).astype(int))

    # Divide the train set into one for each caller, then into 100 sub-groups
    print("Assigning embeddings to caller groups...")
    X_groups = {}
    y_groups = {}
    for c in callers:
        X_groups[c] = np.array_split(X_train[y_train == c], num_groups)
        y_groups[c] = np.array_split(y_train[y_train == c], num_groups)

    # For all callers stack the embeddings of the utterances
    # in a given group, and compute the following:
    # 1. Mean vector (n, )
    # 2. Covariance matrix's diagonal vector (n,)
    print("Computing caller group means and covariances...")
    group_means = {}
    group_diags = {}
    for c in callers:
        group_means[c] = np.array(
            [
                np.mean(np.vstack(X_groups[c][i]), axis=0)
                for i in range(num_groups)
            ]
        )
        group_diags[c] = np.array(
            [
                np.var(np.vstack(X_groups[c][i]), axis=0)
                for i in range(num_groups)
            ]
        )

    # Make 1 gaussian per group for each caller
    print("Making Gaussians per caller group...")
    group_gaussians = {}
    for c in callers:
        group_gaussians[c] = [
            Gaussian(mean=mean, cov=diag)
            for mean, diag in zip(group_means[c], group_diags[c])
        ]

    # Compute the intra group KL-distances within each caller
    print("Computing intra-caller group distances...")
    intra_kl_dist, intra_bhatt_dist = compute_intra_group_distances(
        group_gaussians, callers, num_groups
    )

    # Save intra distances
    print("Saving intra distances...")
    a = "_sub" if args.downsample else ""
    final_save_path = os.path.join(
        save_path, args.expname + f"_intra_distances{a}.pkl"
    )
    save_dict = {"kl_dist": intra_kl_dist, "bhatt_dist": intra_bhatt_dist}
    utils.dump_features(save_dict, final_save_path)

    # Compute KL-distances across callers
    print("Computing inter-callers group distances...")
    inter_kl_dist, inter_bhatt_dist = compute_inter_group_distances(
        group_gaussians, callers, num_groups
    )

    # Save inter distances
    print("Saving inter distances...")
    a = "_sub" if args.downsample else ""
    final_save_path = os.path.join(
        save_path, args.expname + f"_inter_distances{a}.pkl"
    )
    save_dict = {"kl_dist": inter_kl_dist, "bhatt_dist": inter_bhatt_dist}
    utils.dump_features(save_dict, final_save_path)

    print("Done!")
