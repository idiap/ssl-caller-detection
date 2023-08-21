# Copyright (c) 2022, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>
#
# This file converts the extracted features from pre-trained models into fixed-length functionals using mean and std averaging.

import argparse
import glob
import os

import torch

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
        choices=["calltypeID", "marmosetID"],
        help="""
        Call-type classification (`calltypeID`) or caller recognition (`marmosetID`).")
        """,
        required=True,
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


def load_features(features_file):
    import pickle

    with open(features_file, "rb") as handle:
        features_dict = pickle.load(handle)
    return features_dict


def dump_features(features_dict, features_file):
    import pickle

    with open(features_file, "wb") as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    data_path = os.path.join(
        args.expdir, "no-downsample/*/*", args.task, "*.pkl"
    )
    pkl_list = glob.glob(data_path)  # Read all pickle files in a list

    # Iterate over all pickle files - each file contains embeddings of that caller
    for i, pkl in enumerate(tqdm(pkl_list)):
        pkl_id = str(i + 1)

        # Read pickle file containing embedding
        features_dict = load_features(pkl)

        # Iterate
        functionals_dict = {}
        for uid, rep in features_dict.items():  # Iterate over utterances
            layer_list = []
            for l in range(len(rep)):  # Iterate over layers of the utterance
                # data[uid][l].shape is (num_frames?, embedding_length)
                # We average across the num_frames, so the length will be (1, embedding_length)
                # We average for mean and std, and then concatenate both vectors to get a final
                # functional size of (1, 2 * embedding_length)

                means = torch.mean(rep[l], dim=0)
                stds = torch.std(rep[l], dim=0)
                joint_functional = torch.cat((means, stds), dim=0)

                # Append to list, which will contains all layers for this uid utterance
                layer_list.append(joint_functional)

            # Save list with all layers to dict with uid as key
            functionals_dict[uid] = layer_list

        # Save
        save_path = os.path.join(args.savedir, args.expname)
        os.makedirs(save_path, exist_ok=True)
        dump_features(
            functionals_dict,
            os.path.join(save_path, args.expname + "_" + pkl_id + ".pkl"),
        )
