# Copyright (c) 2022, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar
#
# This file sorts the individual extracted embeddings or features into pickle files by caller ID.
# It also computes the functionals and saves them.


import argparse
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

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
        "-g",
        "--groups",
        default=100,
        help="""
        Number of groups to split the train data into to make Gaussians.")
        """,
    )
    parser.add_argument(
        "--savedir_funcs",
        help="""
        Path where to save the functional tensors.")
        """,
    )
    parser.add_argument(
        "--savedir_embeds",
        help="""
        Path where to save the embedding tensors.")
        """,
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=["calltypeID", "marmosetID"],
        help="""
        Call-type classification (`calltypeID`) or caller recognition (`marmosetID`).")
        """,
        required=True,
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
    parser.add_argument(
        "--set",
        required=True,
        choices=["train", "val", "test"],
        help="Select set",
    )
    return parser


def standardise_data(data):
    """
    Standardise the data using sklearn.preprocessing.StandardScaler.
    """
    print("Standardizing embeddings...")

    # v-stack the X data and use sklearn.preprocessing.StandardScaler
    # to fit_transform the data, then un-stack it again
    X = np.vstack(data)

    # Fit and transform the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split X back into a np.array of np.arrays with their original shapes
    X = np.split(X, np.cumsum([len(x) for x in data])[:-1])

    return X


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    # Task
    sset = args.set
    print("sets", sset)
    task = args.task

    # Read with pandas and get length of each list
    train_size = pd.read_csv(
        "marmoset_lists/marmosetID_train.list", header=None
    ).values.shape[0]
    val_size = pd.read_csv(
        "marmoset_lists/marmosetID_val.list", header=None
    ).values.shape[0]
    test_size = pd.read_csv(
        "marmoset_lists/marmosetID_test.list", header=None
    ).values.shape[0]

    # Save path
    if args.downsample:
        ds_path = "downsample"
    else:
        ds_path = "no-downsample"

    # Save path
    save_path_funcs = os.path.join(
        args.savedir_funcs, ds_path, sset, args.expname, task
    )
    save_path_embeds = os.path.join(
        args.savedir_embeds, ds_path, sset, args.expname, task
    )
    os.makedirs(save_path_funcs, exist_ok=True)
    os.makedirs(save_path_embeds, exist_ok=True)

    # Load the mapping between uids and pickleids
    uids_to_pickleid = utils.load_features(
        "marmoset_lists/uids_to_pickleid.pkl"
    )
    pickleid_to_uids = utils.load_features(
        "marmoset_lists/pickleid_to_uids.pkl"
    )

    # Read embeddings
    print("Reading embeddings...")
    X, y = utils.load_baselines(  # Change back to load_embeddings !!
        args.expdir,
        args.expname + "_feats",
        task,
        uids_to_pickleid,
        [sset],
    )

    # Scale the number of groups in function of the set size
    if sset == "train":
        num_groups = args.groups
    elif sset == "val":
        num_groups = val_size * args.groups // train_size
    elif sset == "test":
        num_groups = test_size * args.groups // train_size
    print(f"Number of groups for {sset}: {num_groups}")

    # Downsample the data
    if args.downsample:
        # Number of samples in the smallest class
        num_samples = min(np.unique(y[f"y_{sset}"], return_counts=True)[1])
        # For each class we downsample to the number of samples in the smallest class
        X, y = utils.downsample_data(X, y, sset, num_samples=num_samples)
    else:
        X = X[f"X_{sset}"]
        y = y[f"y_{sset}"]

    # Standardize the data if needed
    X = standardise_data(X) if args.standardize else X
    # y = y if args.downsample else y[f"y_{sets[0]}"]

    print(f"X.shape: {X.shape}, y.shape: {y.shape}")

    callers = sorted(np.unique(y).astype(int))

    # Divide the train set into one for each caller
    print("Assigning embeddings to callers ...")
    X_callers = {}
    y_callers = {}
    for c in callers:
        X_callers[c] = X[y == c]
        y_callers[c] = y[y == c]

    # Divide the train set then into 100 sub-groups
    print("Assigning embeddings to caller groups...")
    X_groups = {}
    y_groups = {}
    for c in callers:
        X_groups[c] = np.array_split(X_callers[c], num_groups)
        y_groups[c] = np.array_split(y_callers[c], num_groups)

    # For all callers stack the embeddings of the utterances
    # in a given group, and compute the following:
    # 1. Mean vector (n, )
    # 2. Covariance matrix's diagonal vector (n,)
    print("Computing caller group means and variances...")
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

    # Save embeddings X_callers and y_callers per caller
    print("Saving caller groups...")
    for c in callers:
        a = "_sub" if args.downsample else ""
        final_save_path = os.path.join(
            save_path_embeds, args.expname + f"_caller_{c}{a}.pkl"
        )
        save_dict = {"X": X_callers[c], "y": y_callers[c]}
        utils.dump_features(save_dict, final_save_path)

    # Save the means and covariances
    print("Saving caller group means and covariances...")
    for c in callers:
        a = "_sub" if args.downsample else ""
        final_save_path = os.path.join(
            save_path_funcs, args.expname + f"_caller_{c}_funcs{a}.pkl"
        )
        save_dict = {"means": group_means[c], "diags": group_diags[c]}
        utils.dump_features(save_dict, final_save_path)

    print("Done!")
