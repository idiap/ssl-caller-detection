# Copyright (c) 2022, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Apoorv Vyas, Eklavya Sarkar
#
# This file extracts handcrafted features (MFCCs and GFCCs) from the wav files.
# The features are saved in the experiment directory `expdir` in a folder named `expname`.
# The features are saved in a pickle file with the name `picklename`.

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm

import utils

RANDOM_SEED = 42


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Run SVM on given features.")
    parser.add_argument(
        "-b",
        "--baseline",
        type=str,
        help="""
        Name of the baseline feature to extract. Must be `mfcc` or `gfcc`.
        """,
    )
    parser.add_argument(
        "-n",
        "--expname",
        help="""
        Name of the baseline feature to extract. Must be `mfcc` or `gfcc`.
        Also will be the name of the folder name in the experiment
        directory `expdir` where the features are stored.
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
        Path where to save the results.")
        """,
    )
    parser.add_argument(
        "-w",
        "--wavs",
        type=str,
        required=True,
        help="path to the input wav files",
    )
    parser.add_argument(
        "--ids", type=str, required=True, help="path to file with unique ids"
    )
    parser.add_argument("--device", default="cuda", help="model.to(device)")
    parser.add_argument(
        "--npickle",
        type=int,
        default=-1,
        help="Number of utterances in a pickle file",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=5,
        help="Number of utterances in a batch. Should be a factor of npickle",
    )
    parser.add_argument(
        "-f", "--picklename", help="Save the pickle with this basename"
    )
    return parser


def load_wav_and_ids(wav_path, id_path):
    with open(wav_path) as fwav, open(id_path) as fid:
        wav_paths = fwav.readlines()
        wav_ids = fid.readlines()

    wav_input_files = {}
    wav_file_paths = []
    wav_file_ids = []
    for wav, wav_id in zip(wav_paths, wav_ids):
        wav_path = wav.split(" ")[0].strip()
        wav_input_files[wav_id.strip()] = wav_path
        wav_file_paths.append(wav_path)
        wav_file_ids.append(wav_id.strip())
    return wav_input_files, wav_file_paths, wav_file_ids


def plot_features(
    padded_wavs,
    gfccs,
    gfccs_deltas,
    gfccs_deltas_deltas,
    gfccs_stack,
    mfccs,
    mfccs_deltas,
    mfccs_deltas_deltas,
    mfccs_stack,
):
    import librosa
    import librosa.display

    fig, axs = plt.subplots(3, 4, figsize=(20, 5))
    axs[2][0].plot(padded_wavs)
    # axs[0][1].plot(padded_wavs)
    # axs[0][2].plot(padded_wavs)

    librosa.display.specshow(
        mfccs,
        x_axis="time",
        sr=16000,
        ax=axs[0][0],
    )
    axs[0][0].set(title="MFCCS 1-13")

    librosa.display.specshow(
        mfccs_deltas,
        x_axis="time",
        sr=16000,
        ax=axs[0][1],
    )
    axs[0][1].set(title="MFCCS 1-13 Deltas")

    librosa.display.specshow(
        mfccs_deltas_deltas,
        x_axis="time",
        sr=16000,
        ax=axs[0][2],
    )
    axs[0][2].set(title="MFCCS 1-13 Delta-Deltas")

    librosa.display.specshow(
        mfccs_stack,
        x_axis="time",
        sr=16000,
        ax=axs[0][3],
    )
    axs[0][3].set(title="MFCCs Stacked")

    #
    librosa.display.specshow(
        gfccs,
        x_axis="time",
        sr=16000,
        ax=axs[1][0],
    )
    axs[1][0].set(title="GFCCS 1-13")

    librosa.display.specshow(
        gfccs_deltas,
        x_axis="time",
        sr=16000,
        ax=axs[1][1],
    )
    axs[1][1].set(title="GFCCs 1-13 Deltas")

    librosa.display.specshow(
        gfccs_deltas_deltas,
        x_axis="time",
        sr=16000,
        ax=axs[1][2],
    )
    axs[1][2].set(title="GFCCs 1-13 Delta-Deltas")

    librosa.display.specshow(
        gfccs_stack,
        x_axis="time",
        sr=16000,
        ax=axs[1][3],
    )
    axs[1][3].set(title="GFCCs Stacked")

    fig.tight_layout()
    plt.savefig("mfccs_combined.png", dpi=300)


def extract_features(
    expname, baseline, wav_paths, wav_ids, batch_size, layer_id, device="cuda"
):
    num_wavs = len(wav_paths)
    features_dict = {}
    for bid in range(0, num_wavs, batch_size):
        end_id = min(bid + batch_size, num_wavs)
        padded_wavs, wavs_len, ids = get_batch(wav_paths, wav_ids, bid, end_id)
        features_stack_list = []

        import spafe

        from spafe.features.gfcc import gfcc
        from spafe.features.mfcc import mfcc
        from spafe.utils.preprocessing import SlidingWindow

        for i, wav in enumerate(padded_wavs):
            if baseline == "mfcc":
                features = mfcc(
                    wav.numpy(),
                    fs=16000,
                    num_ceps=13,
                    pre_emph=1,
                    pre_emph_coeff=0.97,
                    window=SlidingWindow(0.015, 0.005, "hamming"),
                    nfilts=128,
                    nfft=512,
                    low_freq=0,
                    high_freq=8000,
                    normalize="mvn",
                ).T

            elif baseline == "gfcc":
                features = gfcc(
                    wav.numpy(),
                    fs=16000,
                    num_ceps=13,
                    pre_emph=1,
                    pre_emph_coeff=0.97,
                    window=SlidingWindow(0.015, 0.005, "hamming"),
                    nfilts=128,
                    nfft=512,
                    low_freq=0,
                    high_freq=8000,
                    normalize="mvn",
                ).T

            else:
                print("Baseline can only be mfcc or gfcc")
                raise NotImplementedError

            features_deltas = spafe.utils.cepstral.deltas(features)
            features_deltas_deltas = spafe.utils.cepstral.deltas(
                features_deltas
            )
            features_stack = np.concatenate(
                (features, features_deltas, features_deltas_deltas), axis=0
            )

            features_stack_list.append(features_stack)

        features_stack_list = torch.Tensor(np.array(features_stack_list))

        # hidden_states = features_stack_list.transpose(2, 1)
        hidden_states = features_stack_list
        hidden_states_len = wavs_len
        features_dict = pack_features(
            hidden_states, hidden_states_len, ids, layer_id, features_dict
        )
    return features_dict


def get_batch(wav_paths, wav_ids, start, end):
    def getitem(index):
        import soundfile as sf

        path = wav_paths[index]
        wav, curr_sample_rate = sf.read(path, dtype="float32")
        wav /= np.max(np.abs(wav))

        feats = torch.from_numpy(wav).float()
        return feats

    def collate(wavs, padding_value: int = 0):
        from torch.nn.utils.rnn import pad_sequence

        padded_wavs = pad_sequence(
            wavs, batch_first=True, padding_value=padding_value
        )
        return padded_wavs

    end_id = min(end, len(wav_paths))
    wavs = [getitem(index) for index in range(start, end_id)]
    ids = [wav_ids[index] for index in range(start, end_id)]
    wavs_len = [len(wav) for wav in wavs]
    padded_wavs = collate(wavs)
    return padded_wavs, wavs_len, ids


def unpad(hs, hs_len, index, layer):
    if layer == -1:
        hs_unpadded = []
        for h, lens in zip(hs, hs_len):
            l = lens[index]
            hs_unpadded.append(h[index, :l])
    else:
        l = hs_len[index]
        hs_unpadded = hs[index, :l]
    return hs_unpadded


def pack_features(hidden_states, hidden_states_len, ids, layer, features_dict):
    num_inputs = len(ids)
    for input_id in range(num_inputs):
        wav_id = ids[input_id]
        hs = unpad(hidden_states, hidden_states_len, input_id, layer)
        features_dict[wav_id] = hs
    return features_dict


def setup_directories(args):
    if args.expdir is None:
        args.expdir = f"result/downstream/{args.expname}"
    else:
        args.expdir = f"{args.expdir}/{args.expname}"
    os.makedirs(args.expdir, exist_ok=True)


parser = get_argument_parser()
args = parser.parse_args()


if __name__ == "__main__":
    from tqdm import tqdm

    setup_directories(args)

    #
    device = args.device
    expname = args.expname
    baseline = args.baseline
    random.seed(RANDOM_SEED)

    _, wav_file_paths, wav_file_ids = load_wav_and_ids(args.wavs, args.ids)
    num_wav_files = len(wav_file_ids)

    # Extract baseline features
    file_id = 0
    npickle = args.npickle
    for pid in tqdm(range(0, num_wav_files, npickle)):
        file_id += 1
        start_id = pid
        end_id = min(pid + npickle, num_wav_files)
        wav_files = wav_file_paths[start_id:end_id]
        wav_ids = wav_file_ids[start_id:end_id]

        feature_file = "{}/{}_{}.pkl".format(
            args.expdir, args.picklename, file_id
        )

        # print(f"Extracting features for {feature_file}, path:\n{wav_files}")
        # if os.path.exists(feature_file):
        #     print(f"File {feature_file} already exists. Skipping...")
        #     continue

        features_dict = extract_features(
            expname,
            baseline,
            wav_files,
            wav_ids,
            args.bs,
            device,
        )
        utils.dump_features(features_dict, feature_file)
