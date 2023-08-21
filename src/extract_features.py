# Copyright (c) 2022, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Apoorv Vyas, Tilak Purohit, Eklavya Sarkar
#
# This file extracts the features from the pre-trained s3prl models across layers.

import argparse
import os

import numpy as np
import torch

from s3prl import hub
from s3prl.nn import S3PRLUpstream
from s3prl.util.download import set_dir


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Extract the features from the pre-trained model"
    )
    upstreams = [attr for attr in dir(hub) if attr[0] != "_"]
    parser.add_argument(
        "-u",
        "--upstream",
        help=""
        'Upstreams with "_local" or "_url" postfix need local ckpt (-k) or config file (-g). '
        "Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. "
        "Please check upstream/README.md for details. "
        f"Available options in S3PRL: {upstreams}. ",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        default=-1,
        help=""
        "Layer number to extract features. "
        "0th layer corresponds to the first layer (cnn) "
        "The last layer is identified by (#layers - 1) due to zero indexing."
        "default: -1 (all layers)",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="List the number of layers in the model",
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
    parser.add_argument(
        "-n", "--expname", help="Save experiment at result/downstream/expname"
    )
    parser.add_argument("-p", "--expdir", help="Save experiment at expdir")
    parser.add_argument("--device", default="cuda", help="model.to(device)")
    parser.add_argument(
        "--cache_dir",
        help="The cache directory for pretrained model downloading",
    )
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


def setup_directories(args):
    if args.expdir is None:
        args.expdir = f"result/downstream/{args.expname}"
    else:
        args.expdir = f"{args.expdir}/{args.expname}"

    os.makedirs(args.expdir, exist_ok=True)

    if args.cache_dir is not None:
        try:
            set_dir(args.cache_dir)
        except:
            print("Unable to set the path to -> {} \n".format(args.cache_dir))


def get_model(upstream_model, device="cuda"):
    model = S3PRLUpstream(args.upstream)
    model.to(device)
    model.eval()
    return model


def verify_layer_id(model, layer=-1, device="cuda"):
    wavs = torch.randn(2, 16000 * 2).to(
        device
    )  # We give 2 wavs, of len 32k each
    wavs_len = torch.LongTensor([16000 * 1, 16000 * 2]).to(
        device
    )  # We give 2 lengths, 16k and 32k

    reps, reps_len = model(wavs, wavs_len)
    num_layers = len(reps)

    print("INFO: Rep length", reps_len)
    print("INFO: Total layers in the model", num_layers)
    print("INFO: Layer you choose", layer)

    if layer >= 0:
        assert (
            layer + 1
        ) > num_layers, "Layer to extract features doesn't exist. Select between 0 and {} (inclusive)".format(
            num_layers - 1
        )

    return num_layers


def get_features(model, wavs, wavs_len, layer=-1, device="cuda"):
    with torch.no_grad():
        wavs_len = torch.LongTensor(wavs_len).to(device)
        all_hs, all_hs_len = model(wavs.to(device), wavs_len)

    for layer_id, (hs, hs_len) in enumerate(zip(all_hs, all_hs_len)):
        hs = hs.to("cpu")
        hs_len = hs_len.to("cpu")
        assert isinstance(hs, torch.FloatTensor)
        assert isinstance(hs_len, torch.LongTensor)

        if layer == layer:
            hidden_states = hs
            hidden_states_len = hs_len

        assert hs_len.dim() == 1

    if layer == -1:
        hidden_states = [hs.to("cpu") for hs in all_hs]
        hidden_states_len = [hs_len.to("cpu") for hs_len in all_hs_len]
    return hidden_states, hidden_states_len


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


def extract_features(
    model, wav_paths, wav_ids, batch_size, layer_id, device="cuda"
):
    num_wavs = len(wav_paths)
    features_dict = {}
    for bid in range(0, num_wavs, batch_size):
        end_id = min(bid + batch_size, num_wavs)
        padded_wavs, wavs_len, ids = get_batch(
            wav_file_paths, wav_ids, bid, end_id
        )
        hidden_states, hidden_states_len = get_features(
            model, padded_wavs, wavs_len, layer_id, device
        )
        features_dict = pack_features(
            hidden_states, hidden_states_len, ids, layer_id, features_dict
        )
    return features_dict


def dump_features(features_dict, features_file):
    import pickle

    with open(features_file, "wb") as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_features(features_dict, features_file):
    import pickle

    with open(features_file, "rb") as handle:
        features_dict = pickle.load(handle)
    return features_dict


parser = get_argument_parser()
args = parser.parse_args()


if __name__ == "__main__":
    from tqdm import tqdm

    setup_directories(args)
    device = args.device
    layer_id_for_extraction = args.layer
    model = get_model(args.upstream, device)

    num_layers = verify_layer_id(model, layer_id_for_extraction)
    if args.info:
        print("Total layers are: {}".format(num_layers))
        exit(0)

    wav_input_files, wav_file_paths, wav_file_ids = load_wav_and_ids(
        args.wavs, args.ids
    )
    if args.npickle > 0:
        assert (
            args.npickle % args.bs == 0
        ), "Number of utterances in pickle must be divisible by batchsize."

    num_wav_files = len(wav_file_ids)
    if args.npickle == -1:
        features_dict = extract_features(
            model,
            wav_file_paths,
            wav_file_ids,
            args.bs,
            layer_id_for_extraction,
            device,
        )
        feature_file = "{}/{}.pkl".format(args.expdir, args.picklename)
        dump_features(features_dict, feature_file)
    else:
        file_id = 0
        npickle = args.npickle
        for pid in tqdm(range(0, num_wav_files, npickle)):
            file_id += 1
            start_id = pid
            end_id = min(pid + npickle, num_wav_files)
            wav_files = wav_file_paths[start_id:end_id]
            wav_ids = wav_file_ids[start_id:end_id]
            features_dict = extract_features(
                model,
                wav_files,
                wav_ids,
                args.bs,
                layer_id_for_extraction,
                device,
            )
            feature_file = "{}/{}_{}.pkl".format(
                args.expdir, args.picklename, file_id
            )
            dump_features(features_dict, feature_file)
