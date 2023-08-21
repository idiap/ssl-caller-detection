# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>


import os

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm


def load_features(features_file):
    import pickle

    with open(features_file, "rb") as handle:
        features_dict = pickle.load(handle)
    return features_dict


def dump_features(features_dict, features_file):
    import pickle

    with open(features_file, "wb") as handle:
        # Do not overwrite existing files
        # if os.path.exists(features_file):
        #     print(f"File {features_file} already exists. Skipping...")
        # else:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings(
    expdir, expname, task, uids_to_pickleid, sets=["train", "val", "test"]
):
    """
    Load the embeddings from the pickle files.
    """

    X = {f"X_{s}": [] for s in sets}
    y = {f"y_{s}": [] for s in sets}

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
            f"marmoset_lists/uids_{task}_{s}.list", header=None, names=["uid"]
        ).values.flatten()

        for uid in tqdm(uids[s]):
            # Go from UID to PID
            uid = str(uid)
            pickleid = uids_to_pickleid[uid]

            # Take only the last layer
            embedding_path = os.path.join(
                expdir, expname, expname + f"_{pickleid}.pkl"
            )
            embedding = load_features(embedding_path)[uid][
                selected_layer
            ]  # (n, 256)

            # Append - list of (m, 256) tensors
            X[f"X_{s}"].append(embedding.numpy())

            # Get corresponding label using the wav_list and stack to y_train
            label = wav_lists[s][wav_lists[s]["uid"] == uid]["label"].values[0]
            y[f"y_{s}"].append(label)

        # Convert to numpy arrays
        X[f"X_{s}"] = np.array(X[f"X_{s}"], dtype=object)
        y[f"y_{s}"] = np.array(y[f"y_{s}"])

    return X, y


def load_baselines(
    expdir, expname, task, uids_to_pickleid, sets=["train", "val", "test"]
):
    """
    Load the baselines (eg MFCCs) from the pickle files.
    """

    X = {f"X_{s}": [] for s in sets}
    y = {f"y_{s}": [] for s in sets}

    wav_lists = {}
    uids = {}

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
            f"marmoset_lists/uids_{task}_{s}.list", header=None, names=["uid"]
        ).values.flatten()

        for uid in tqdm(uids[s]):
            # Go from UID to PID
            uid = str(uid)
            pickleid = uids_to_pickleid[uid]

            # Take only the last layer
            baseline_path = os.path.join(
                expdir, expname, expname + f"_{pickleid}.pkl"
            )
            baseline = load_features(baseline_path)[uid].T  # (n, 39)

            # Append - list of (m, 39) tensors
            X[f"X_{s}"].append(baseline.numpy())

            # Get corresponding label using the wav_list and stack to y_train
            label = wav_lists[s][wav_lists[s]["uid"] == uid]["label"].values[0]
            y[f"y_{s}"].append(label)

        # Convert to numpy arrays
        X[f"X_{s}"] = np.array(X[f"X_{s}"], dtype=object)
        y[f"y_{s}"] = np.array(y[f"y_{s}"])

    return X, y


def load_functionals(
    expdir, expname, task, uids_to_pickleid, sets=["train", "val", "test"]
):
    X = {f"X_{s}": torch.tensor([]) for s in sets}
    y = {f"y_{s}": torch.tensor([]) for s in sets}

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

        for uid in tqdm(uids[s]):
            uid = str(uid)
            pickleid = uids_to_pickleid[uid]

            # Take only the last layer and vstack all the functionals into a single tensor
            functional_path = os.path.join(
                expdir, expname, expname + f"_{pickleid}.pkl"
            )
            functionals = load_features(functional_path)[uid][
                selected_layer
            ].unsqueeze(dim=0)

            # V-append the functional
            x_set_name = "X_" + s
            X[f"X_{s}"] = torch.cat((X[f"X_{s}"], functionals), dim=0)

            # Get corresponding label using the wav_list and stack to y_train
            label = wav_lists[s][wav_lists[s]["uid"] == uid]["label"].values[0]
            y[f"y_{s}"] = torch.cat((y[f"y_{s}"], torch.tensor([label])))

        # Convert to numpy arrays
        X[f"X_{s}"] = X[f"X_{s}"].numpy()
        y[f"y_{s}"] = y[f"y_{s}"].numpy()

    return X, y


def downsample_data(X, y, sset, num_samples):
    """
    Down-sample the data using the labels.
    """
    from sklearn.utils import resample

    classes = [*range(1, 11)]

    print("Down-sampling data for each class randomly ...")
    # Downsample to the same number of samples per class
    X_downsampled = []
    y_downsampled = []
    for c in classes:
        # Get the indices of the class
        idx = np.where(y[f"y_{sset}"] == c)[0]

        # Select the samples of the class
        X_ds = X[f"X_{sset}"][idx]
        y_ds = y[f"y_{sset}"][idx]

        # Downsample
        X_ds, y_ds = resample(
            X_ds, y_ds, replace=False, n_samples=num_samples, random_state=42
        )

        # Append
        X_downsampled.append(X_ds)
        y_downsampled.append(y_ds)

    # Concatenate
    X_downsampled = np.concatenate(X_downsampled, axis=0)
    y_downsampled = np.concatenate(y_downsampled, axis=0)

    return X_downsampled, y_downsampled


def load_functionals_covar(expdir, ds_path, sets, expname, task, callers):
    """
    Load the functionals from the pickle files.
    """
    funcs_dict = {s: {} for s in sets}
    labels = {s: {} for s in sets}
    app = "" if "no" in ds_path else "_sub"
    for c in callers:
        for s in sets:
            features = load_features(
                os.path.join(
                    expdir,
                    ds_path,
                    s,
                    expname,
                    task,
                    expname + f"_caller_{c}_funcs{app}.pkl",
                )
            )
            funcs_dict[s][c] = np.concatenate(
                (features["means"], features["diags"]), axis=1
            )
            labels[s][c] = np.ones((features["means"].shape[0],)) * c
    return funcs_dict, labels
