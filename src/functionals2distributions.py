# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>


import argparse
import os

import numpy as np
import pandas as pd
import scipy
import torch

from kaldi_GMM import Gaussian
from tqdm import tqdm


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


def load_features(features_file):
    import pickle

    with open(features_file, "rb") as handle:
        features_dict = pickle.load(handle)
    return features_dict


def dump_features(features_dict, features_file):
    import pickle

    with open(features_file, "wb") as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_features_and_labels(expdir, expname, task):
    LIMIT = 10000

    X = {
        "X_train": torch.tensor([]),
        "X_val": torch.tensor([]),
        "X_test": torch.tensor([]),
    }

    y = {
        "y_train": torch.tensor([]),
        "y_val": torch.tensor([]),
        "y_test": torch.tensor([]),
    }

    sets = ["train", "val", "test"]
    wav_lists = {}
    uids = {}
    selected_layer = -1

    for s in sets:
        print(f"Processing {s} set...")

        # Read lists
        wav_lists[s] = pd.read_csv(
            f"marmoset_lists/{task}_{s}.list",
            header=None,
            names=["path", "label"],
            sep=" ",
        )

        # Add the uid column
        wav_lists[s]["uid"] = wav_lists[s]["path"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )

        # Read uids into a dict
        uids[s] = pd.read_csv(
            f"marmoset_lists/uids_{task}_{s}.list",
            header=None,
            names=["uid"],
        ).values.flatten()

        if s != "train":
            LIMIT = LIMIT // 5
        else:
            # Randomise
            uids[s] = uids[s][torch.randperm(len(uids[s]))]

        for uid in tqdm(uids[s][:LIMIT]):
            uid = str(uid)
            pickleid = uids_to_pickleid[uid]

            # Take only the last layer and vstack all the functionals into a single tensor
            functionals = load_features(
                os.path.join(expdir, expname, expname + f"_{pickleid}.pkl")
            )[uid][selected_layer].unsqueeze(dim=0)

            # V-append the functional
            x_set_name = "X_" + s
            X[x_set_name] = torch.cat((X[x_set_name], functionals), dim=0)

            # Get corresponding label using the wav_list and stack to y_train
            y_set_name = "y_" + s
            label = wav_lists[s][wav_lists[s]["uid"] == uid]["label"].values[0]
            y[y_set_name] = torch.cat((y[y_set_name], torch.tensor([label])))

        # Print shapes
        print(s, X[x_set_name].shape, y[y_set_name].shape)

    return X, y


def convert_from_torch_to_numpy(X, y):
    for x_set_name in X.keys():
        X[x_set_name] = X[x_set_name].numpy()

    for y_set_name in y.keys():
        y[y_set_name] = y[y_set_name].numpy()

    return X, y


def compute_bhattacharya_distance(p, q, return_coeff=True):
    """
    Compute the Bhattacharya coefficient and distance between two distributions.
    """
    bc_coefficient = np.sum(np.sqrt(p * q))
    bc_distance = -np.log(bc_coefficient)
    if np.isinf(bc_distance):
        bc_distance = 0
    if return_coeff:
        return bc_distance, bc_coefficient
    else:
        return bc_distance


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    # Task
    task = args.task
    groups = args.groups

    # Save path
    save_path = os.path.join(args.savedir, args.expname, args.task)
    os.makedirs(save_path, exist_ok=True)

    # Load the mapping between uids and pickleids
    uids_to_pickleid = load_features("marmoset_lists/uids_to_pickleid.pkl")
    pickleid_to_uids = load_features("marmoset_lists/pickleid_to_uids.pkl")

    # Read functionals
    print("Reading functionals...")
    X, y = load_features_and_labels(
        args.expdir, args.expname.replace("distris", "funcs"), task
    )

    # Convert to numpy
    X, y = convert_from_torch_to_numpy(X, y)

    # Only use the train set
    X_train = X["X_train"]
    y_train = y["y_train"]
    callers = sorted(np.unique(y_train).astype(int))

    # Statistics
    # Split at the middle of functionals for means and stds
    split = X_train.shape[-1] // 2
    x_min = X_train.min()
    x_max = X_train.max()

    # Divide the train set into one for each caller
    X_train_by_caller = {}
    y_train_by_caller = {}
    for c in callers:
        X_train_by_caller[c] = X_train[y_train == c]
        y_train_by_caller[c] = y_train[y_train == c]

    # Divide into 100 sub-grounds, and compute their mean and std
    X_train_by_group = {}
    y_train_by_group = {}

    for c in callers:
        X_train_by_group[c] = np.array_split(X_train_by_caller[c], groups)
        y_train_by_group[c] = np.array_split(y_train_by_caller[c], groups)

    # Compute one mean vector (n, )and covariance matrix (n,n) for each group,
    # where n is the means-std functional dimension divided by 2.
    X_train_by_group_mean = {}
    X_train_by_group_cov = {}

    for c in callers:
        # Calculate an average mean vector per group
        X_train_by_group_mean[c] = np.array(
            [
                np.mean(X_train_by_group[c][i][:, :split], axis=0)
                for i in range(groups)
            ]
        )

        # Calculate an average std vector per group, and make a diagonal cov matrix with it
        average_stds = [
            np.mean(X_train_by_group[c][i][:, split:], axis=0)
            for i in range(groups)
        ]

        X_train_by_group_cov[c] = np.array(
            [np.diag(average_stds[i]) for i in range(groups)]
        )

    # print("Mean:", X_train_by_group_mean[1][0].shape)
    # print("Cov:", X_train_by_group_cov[1][0].shape)

    # Make Gaussians for each group
    X_train_by_group_gaussians = {}
    X_train_by_group_gaussians_kaldi = {}

    # For each caller
    for c in callers:
        # Make 1 gaussian per group -> 100 gaussians per group
        X_train_by_group_gaussians[c] = [
            scipy.stats.multivariate_normal(mean, cov)
            for mean, cov in zip(
                X_train_by_group_mean[c], X_train_by_group_cov[c]
            )
        ]

        X_train_by_group_gaussians_kaldi[c] = [
            Gaussian(mean=mean, cov=cov)
            for mean, cov in zip(
                X_train_by_group_mean[c], X_train_by_group_cov[c]
            )
        ]

    # Compute the inter group KL-distances within each caller
    inter_group_kl_distances = {}
    inter_group_kl_distances_kaldi = {}
    inter_group_bhatt_coeff = {}
    inter_group_bhatt_dist = {}

    # Iterate over callers
    print("Computing inter-group distances...")
    for c in tqdm(callers):
        inter_group_kl_distances[c] = []
        inter_group_kl_distances_kaldi[c] = []
        inter_group_bhatt_coeff[c] = []
        inter_group_bhatt_dist[c] = []

        # Iterate over groups within each caller with a double index
        for i in range(groups):
            for j in range(groups):
                # We don't want distances between the same group
                if i != j:
                    # Compute the pdf of the two selected groups
                    pdf_i = X_train_by_group_gaussians[c][i].pdf(
                        X_train[:, :split][:1000]
                    )

                    pdf_j = X_train_by_group_gaussians[c][j].pdf(
                        X_train[:, :split][:1000]
                    )

                    # Compute the KL distance between the two selected groups
                    inter_group_kl_distances[c].append(
                        scipy.stats.entropy(pdf_i, pdf_j, base=None, axis=0)
                    )

                    # Enno's kaldi script
                    g1 = X_train_by_group_gaussians_kaldi[c][i]
                    g2 = X_train_by_group_gaussians_kaldi[c][j]
                    inter_group_kl_distances_kaldi[c].append(
                        Gaussian.D_KL(g1, g2)
                    )

                    # Calculate the bhattacharyya distance between the two groups
                    bc_dist, bc_coeff = compute_bhattacharya_distance(
                        pdf_i, pdf_j, return_coeff=True
                    )

                    # Save
                    inter_group_bhatt_coeff[c].append(bc_coeff)
                    inter_group_bhatt_dist[c].append(bc_dist)

    # # Save distances
    print("Saving inter distances...")
    final_save_path = os.path.join(
        save_path, args.expname + "_inter_distances.pkl"
    )
    dump_features(
        {
            "inter_group_kl_distances": inter_group_kl_distances,
            "inter_group_bhatt_coeff": inter_group_bhatt_coeff,
            "inter_group_bhatt_dist": inter_group_bhatt_dist,
        },
        final_save_path,
    )

    # Compute KL-distances across callers
    across_callers_kl_distances = {}
    across_callers_bhatt_coeff = {}
    across_callers_bhatt_dist = {}

    # Iterate over callers with double indices
    print("Computing across-callers distances...")
    for c_i in tqdm(callers):
        for c_j in callers:
            # We don't want distances between the same caller
            if c_i != c_j:
                across_callers_kl_distances[(c_i, c_j)] = []
                across_callers_bhatt_coeff[(c_i, c_j)] = []
                across_callers_bhatt_dist[(c_i, c_j)] = []

                # Iterate over groups within the two chosen callers with a double index
                for i in range(groups):
                    for j in range(groups):
                        # We don't want distances between the same group
                        if i != j:
                            # Compute the pdf of the two selected groups
                            pdf_ci_i = X_train_by_group_gaussians[c_i][i].pdf(
                                X_train[:, :split][:1000]
                            )

                            pdf_cj_j = X_train_by_group_gaussians[c_j][j].pdf(
                                X_train[:, :split][:1000]
                            )

                            # Compute the KL distance between the two groups
                            across_callers_kl_distances[(c_i, c_j)].append(
                                scipy.stats.entropy(
                                    pdf_ci_i, pdf_cj_j, base=None, axis=0
                                )
                            )

                            # Calculate the bhattacharyya distance between the two groups
                            bc_dist, bc_coeff = compute_bhattacharya_distance(
                                pdf_i, pdf_j, return_coeff=True
                            )

                            # Save
                            inter_group_bhatt_coeff[c].append(bc_coeff)
                            inter_group_bhatt_dist[c].append(bc_dist)

    # # Save distances
    final_save_path = os.path.join(
        save_path, args.expname + "_intra_distances.pkl"
    )
    dump_features(
        {
            "across_callers_kl_distances": across_callers_kl_distances,
            "across_callers_bhatt_coeff": across_callers_bhatt_coeff,
            "across_callers_bhatt_dist": across_callers_bhatt_dist,
        },
        final_save_path,
    )
