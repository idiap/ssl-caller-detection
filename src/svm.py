# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>


import argparse
import json
import os

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import recall_score

import utils


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Run SVM on given features.")
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
        Path where to save the results.")
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


def convert_from_torch_to_numpy(X, y):
    for x_set_name in X.keys():
        X[x_set_name] = X[x_set_name].numpy()

    for y_set_name in y.keys():
        y[y_set_name] = y[y_set_name].numpy()

    return X, y


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    # Task
    task = args.task

    # Save path
    save_path = os.path.join(args.savedir, args.expname, args.task)
    os.makedirs(save_path, exist_ok=True)

    # Load the mapping between uids and pickleids
    uids_to_pickleid = utils.load_features(
        "marmoset_lists/uids_to_pickleid.pkl"
    )
    pickleid_to_uids = utils.load_features(
        "marmoset_lists/pickleid_to_uids.pkl"
    )

    # Load the mapping between calltypes and indices
    calltype_to_index_path = "dataset/calltype_to_index_reduced.json"

    # Labels
    if task == "calltypeID":
        with open(calltype_to_index_path) as f:
            target_names = json.load(f)
    else:
        target_names = None

    # Read functionals
    print("Reading functionals...")
    X, y = utils.load_functionals(
        args.expdir, args.expname + "_funcs", task, uids_to_pickleid
    )

    # # Convert to numpy arrays
    # X, y = convert_from_torch_to_numpy(X, y)

    # Downsample the data
    # if args.downsample:
    #     X_train, y_train = utils.downsample_data(X, y, num_samples=2681)

    # Train the SVM
    print("Training the SVM...")
    clf = svm.SVC()
    clf.fit(X["X_train"], y["y_train"])

    # Save the model
    print("Saving the SVM...")
    final_save_path = os.path.join(save_path, args.expname + "_clf.pkl")
    utils.dump_features(clf, final_save_path)

    # Test the SVM
    print("Testing the SVM...")
    preds_train = clf.predict(X["X_train"])
    preds_val = clf.predict(X["X_val"])
    preds_test = clf.predict(X["X_test"])

    # Get classification_report
    print("Getting metrics...")
    report_train = classification_report(
        y["y_train"], preds_train, target_names=target_names, output_dict=True
    )
    print(pd.DataFrame.from_dict(report_train).T.round(2))
    final_save_path = os.path.join(save_path, args.expname + "_train.pkl")
    utils.dump_features(report_train, final_save_path)

    report_val = classification_report(
        y["y_val"], preds_val, target_names=target_names, output_dict=True
    )
    print(pd.DataFrame.from_dict(report_val).T.round(2))
    final_save_path = os.path.join(save_path, args.expname + "_val.pkl")
    utils.dump_features(report_val, final_save_path)

    report_test = classification_report(
        y["y_test"], preds_test, target_names=target_names, output_dict=True
    )
    print(pd.DataFrame.from_dict(report_test).T.round(2))
    final_save_path = os.path.join(save_path, args.expname + "_test.pkl")
    utils.dump_features(report_test, final_save_path)

    # Get accuracy_score
    print("Acc train:", accuracy_score(y["y_train"], clf.predict(X["X_train"])))
    print("Acc val:", accuracy_score(y["y_val"], clf.predict(X["X_val"])))
    print("Acc test:", accuracy_score(y["y_test"], clf.predict(X["X_test"])))

    # Get TP, FP, TN, FN
    conf_matrix = cm(y["y_test"], preds_test)
    tn, fp, fn, tp = conf_matrix.ravel()
    print("tn, fp, fn, tp:", tn, fp, fn, tp)

    # Calculate accuracy and F1 score
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    uar = recall_score(y["y_test"], preds_test, average="macro")

    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("uar:", uar)

    # Make a confusion matrix
    print("Making a confusion matrix...")
    conf_matrix = (
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    )
    conf_matrix = pd.DataFrame(
        conf_matrix, index=target_names, columns=target_names
    )
    conf_matrix = conf_matrix.round(2)
    print(conf_matrix)
    final_save_path = os.path.join(save_path, args.expname + "_conf_matrix.pkl")
    utils.dump_features(conf_matrix, final_save_path)
