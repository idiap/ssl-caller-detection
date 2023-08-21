# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import argparse
import glob
import json
import os
import random

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import auc

import utils


RANDOM_SEED = 42


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
        choices=["calltypeID", "marmosetID"],
        help="""
        Call-type classification (`calltypeID`) or caller recognition (`marmosetID`).")
        """,
        required=True,
    )

    return parser


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
        "mfcc": "MFCCs",
        "gfcc": "GFCCs",
    }

    classifier_names = {
        "LinearSVC": "L_SVM",
        "SVC": "SVM",
        "RF": "RF",
        "AB": "AB",
    }

    return model_names, classifier_names


def randomColorGenerator(number_of_colors=1, seed=0):
    """Generate list of random colors"""
    np.random.seed(seed)
    return [
        "#" + "".join([np.random.choice(list("0123456789ABCDEF")) for j in range(6)])
        for i in range(number_of_colors)
    ]


def save_macro_aucs(expdir, task, m_names, c_names):
    metrics_path = os.path.join(expdir, "*/*", task, "*", "*_all_folds_metrics.pkl")
    metrics_pkl_list = glob.glob(metrics_path)

    # Read all metrics in a pandas dataframe
    df = pd.DataFrame()
    rows = []

    # Iterate over metrics pickle files
    for metrics_pkl in metrics_pkl_list:
        # Load pickle
        pkl = utils.load_features(metrics_pkl)

        # Get average AUCs across all folds
        avg_train_auc = np.mean([pkl[f]["train"]["auc"]["macro"] for f in pkl.keys()])
        avg_val_auc = np.mean([pkl[f]["val"]["auc"]["macro"] for f in pkl.keys()])
        avg_test_auc = np.mean([pkl[f]["test"]["auc"]["macro"] for f in pkl.keys()])
        avg_aucs = [avg_train_auc, avg_val_auc, avg_test_auc]

        # Get average UAR across all folds
        avg_train_uar = np.mean([pkl[f]["train"]["uar"] for f in pkl.keys()])
        avg_val_uar = np.mean([pkl[f]["val"]["uar"] for f in pkl.keys()])
        avg_test_uar = np.mean([pkl[f]["test"]["uar"] for f in pkl.keys()])
        avg_uars = [avg_train_uar, avg_val_uar, avg_test_uar]

        # Get average Macro-F1 across all folds
        avg_train_f1 = np.mean([pkl[f]["train"]["f1_score_macro"] for f in pkl.keys()])
        avg_val_f1 = np.mean([pkl[f]["val"]["f1_score_macro"] for f in pkl.keys()])
        avg_test_f1 = np.mean([pkl[f]["test"]["f1_score_macro"] for f in pkl.keys()])
        avg_f1s = [avg_train_f1, avg_val_f1, avg_test_f1]

        # Get average eer_macro across all folds
        avg_train_eer = np.mean([pkl[f]["train"]["eer_macro"] for f in pkl.keys()])
        avg_val_eer = np.mean([pkl[f]["val"]["eer_macro"] for f in pkl.keys()])
        avg_test_eer = np.mean([pkl[f]["test"]["eer_macro"] for f in pkl.keys()])
        avg_eers = [avg_train_eer, avg_val_eer, avg_test_eer]

        # Get model name
        dataset = metrics_pkl.split("/")[-5]
        model_name = metrics_pkl.split("/")[-4]
        model_name_red = m_names[model_name]
        classifier = c_names[metrics_pkl.split("/")[-2]]
        model_info = [model_name_red, dataset, classifier]

        # Append to rows
        rows.append(model_info + avg_aucs + avg_uars + avg_f1s + avg_eers)

    # Make dataframe from rows
    aucs = ["train_auc", "val_auc", "test_auc"]
    uars = ["train_uar", "val_uar", "test_uar"]
    f1s = ["train_f1", "val_f1", "test_f1"]
    eers = ["train_eer", "val_eer", "test_eer"]
    columns = ["model", "dataset", "classifier"] + aucs + uars + f1s + eers
    df = pd.DataFrame(rows, columns=columns)

    # Make pivot table
    df_pivot = df.pivot_table(
        index=["model", "dataset"],
        columns="classifier",
        values=aucs + uars + f1s + eers,
    )

    # Show the test table
    test_metrics = ["test_auc", "test_f1", "test_uar", "test_eer"]
    test_df = df_pivot.xs("no-downsample", level="dataset", axis=0)[test_metrics]
    test_df_ds = df_pivot.xs("downsample", level="dataset", axis=0)[test_metrics]

    # Get a view with only SVM
    svm_df = test_df.xs("SVM", level="classifier", axis=1)
    svm_df_ds = test_df_ds.xs("SVM", level="classifier", axis=1)

    # Add an average column for test_df AUC scores
    # test_df["average_auc"] = test_df["test_auc"].mean(axis=1)

    # Add an average row for df_pivot_test.
    # test_df_nods = test_df.copy()
    # test_df_nods.loc["Average"] = test_df_nods.mean()

    # Save path
    final_save_path = os.path.join(
        args.savedir, f"{args.task}_callergroup_classif_results.tex"
    )

    final_save_path_ds = os.path.join(
        args.savedir, f"{args.task}_callergroup_classif_results_ds.tex"
    )

    # Save
    (test_df * 100).to_latex(final_save_path, float_format="{:0.2f}".format)
    (test_df * 100).to_csv(final_save_path.replace(".tex", ".csv"))

    (test_df_ds * 100).to_latex(final_save_path_ds, float_format="{:0.2f}".format)
    (test_df_ds * 100).to_csv(final_save_path_ds.replace(".tex", ".csv"))

    # save svm_df
    (svm_df * 100).to_latex(
        final_save_path.replace(".tex", "_svm.tex"),
        float_format="{:0.2f}".format,
    )
    (svm_df * 100).to_csv(final_save_path.replace(".tex", "_svm.csv"))

    (svm_df_ds * 100).to_latex(
        final_save_path_ds.replace(".tex", "_svm.tex"),
        float_format="{:0.2f}".format,
    )
    (svm_df_ds * 100).to_csv(final_save_path_ds.replace(".tex", "_svm.csv"))

    # Make another one where the classifiers are the index
    df_pivot2 = df.pivot_table(
        index=["model", "classifier"],
        columns="dataset",
        values=["train_auc", "val_auc", "test_auc"],
    )


def make_fpr_tpr_auc_dicts(y, y_ohe, probs_list, class_names):
    """
    Compute and return the ROC curve and ROC area for each class in dictionaries.
    Macro-average ROC curve, which gives equal weight to the classification of each class.
    Micro-average ROC curve, which gives equal weight to the classification of each sample.
    AUC is the area under the ROC curve, which gives a measure of separability and
    tell us the strength of classification rates in numbers.
    """
    # Dicts
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    auc_metric = dict()
    num_classes = len(class_names)

    # Calculate ROC curve and ROC area for each class in 'one vs all' scenario
    for i in range(num_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(
            y_ohe[:, i], probs_list[:, i], pos_label=1
        )
        auc_metric[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area.
    fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(
        y_ohe.ravel(), probs_list.ravel()
    )
    auc_metric["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # 1. First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # 2. Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # 3. Finally average it and compute AUC
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc_metric["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, thresholds, auc_metric


def read_distance_pkls(expdir, task, m_names, c_names):
    distances_path = os.path.join(
        "/path/to/dataset/distributions",
        "*",
        task,
        "*_distances.pkl",
    )
    distances_pkl_list = glob.glob(distances_path)

    # Read pickles into a dataframe
    df = pd.DataFrame()
    rows = []

    for distances_pkl in distances_pkl_list:
        # Get model name
        model_name = distances_pkl.split("/")[-3].replace("_distris", "")
        model_name_red = m_names[model_name]
        dist_type = distances_pkl.split("/")[-1].split("_")[-2]
        x_name = "KL" if dist_type == "kl_dist" else "BC"

        # Load pickle
        distances = utils.load_features(distances_pkl)

        # Compile distances on a table
        for c, dist in distances[dist_type].items():
            rows.append([model_name_red, c, x_name, dist])

    # Make dataframe from rows
    columns = ["model", "classifier", "x_name", "distance"]
    df = pd.DataFrame(rows, columns=columns)


def plot_roc_curves(expdir, task, m_names, c_names, num_classes, ds_type, classifier):
    """
    Plot the ROC AUC curves for each model.
    For each model:
        For each fold:
            For test set:
                Plot macro ROC AUC curves.
    """
    metrics_path = os.path.join(
        expdir, ds_type, "*", task, classifier, "*_all_folds_metrics.pkl"
    )
    metrics_pkl_list = glob.glob(metrics_path)
    class_colors = randomColorGenerator(11, RANDOM_SEED)
    cache = {}

    # Make plot
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Iterate over metrics pickle files
    for metrics_pkl in metrics_pkl_list:
        # Load pickle
        pkl = utils.load_features(metrics_pkl)

        # Get model info
        dataset = metrics_pkl.split("/")[-5]
        model_name = metrics_pkl.split("/")[-4]
        model_name_red = m_names[model_name]
        # classifier = c_names[metrics_pkl.split("/")[-2]]
        model_info = [model_name_red, dataset, classifier]

        if model_name_red not in cache:
            cache[model_name_red] = True

            # Get average tpr values across all folds, then average across all classes
            tprs_all = []
            aucs_all = []
            mean_fpr = np.linspace(0, 1, 100)

            # For each fold
            for f in pkl.keys():
                tprs = []
                aucs = []

                # For each class
                for i in range(num_classes):
                    # Get fpr and tpr values
                    fpr_f_i = pkl[f]["test"]["fpr"][i]
                    tpr_f_i = pkl[f]["test"]["tpr"][i]

                    # Interpolate
                    tprs.append(np.interp(mean_fpr, fpr_f_i, tpr_f_i))
                    tprs[-1][0] = 0.0

                    roc_auc = auc(fpr_f_i, tpr_f_i)
                    aucs.append(roc_auc)

                # Now we have tprs for all classes for this fold
                # Therefore we can get mean tpr and auc values across all classes for this fold
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)

                tprs_all.append(mean_tpr)
                aucs_all.append(mean_auc)

                # Plot the fold
                # ax.plot(
                #     mean_fpr,
                #     mean_tpr,
                #     lw=2,
                #     alpha=0.8,
                #     label="Fold %d (AUC = %0.2f)" % (f, mean_auc),
                # )

            # Now we have tprs for all classes for all folds
            # Therefore we can get mean tpr and auc values across all classes and folds
            mean_tpr = np.mean(tprs_all, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs_all)
            std_auc = np.std(aucs_all)

            # Plot the mean ROC AUC curve for this model
            ax.plot(
                mean_fpr,
                mean_tpr,
                # color="b",
                # label=model_name_red,
                label=f"{model_name_red} Mean (AUC = {np.round(mean_auc, 2)} $\pm$ {np.round(std_auc, 2)})",
                lw=2,
                alpha=0.8,
            )

            # Plot the +- 1 std
            std_tpr = np.std(tprs_all, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                # color="grey",
                alpha=0.2,
                # label=r"$\pm$ 1 std. dev.",
            )

    # Plot the random guessing line
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.grid(True, linestyle="dotted", alpha=1)
    ax.legend(loc="lower right")

    # Set axis labels and title
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # ax.set_title(f"{classifier} ROC curves on Test")
    # Make a legend outside at the bottom
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)

    os.makedirs("dump")
    plt.savefig(f"dump/{classifier}_roc_auc_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"dump/{classifier}_roc_auc_curves.png", dpi=300, bbox_inches="tight")


def plot_roc_auc_curves(
    pkl,
    fold_id,
    model_name,
    num_classes,
    savedir,
    expname,
    split,
    classifier,
    seed=RANDOM_SEED,
):
    """Plot ROC AUC Curves"""

    # Get the fpr, tpr, and auc for the specified model, fold, and split
    fpr = pkl[fold_id][split]["fpr"]
    tpr = pkl[fold_id][split]["tpr"]
    auc_metric = pkl[fold_id][split]["auc"]

    fig, ax = plt.subplots(dpi=150, figsize=(5, 5))

    lw = 2
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    class_colors = randomColorGenerator(num_classes, seed)

    # Plot ROC curve for each class
    for i in range(num_classes):
        ax.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            color=class_colors[i],
            label="{0} ({1:0.2f}%)" "".format(str(i + 1), auc_metric[i] * 100),
        )

    # Plot micro and macro curves
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        lw=lw,
        label="Micro avg ({:0.2f}%)" "".format(auc_metric["micro"] * 100),
        linestyle=":",
        color="deeppink",
    )
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        lw=lw,
        label="Macro avg ({:0.2f}%)" "".format(auc_metric["macro"] * 100),
        linestyle=":",
        color="navy",
    )

    print(
        f"{model_name}, {classifier}, set {split}, fold {fold_id+1}, FPR: {fpr['macro'].shape}, TPR: {tpr['macro'].shape}"
    )

    # Baseline
    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    # Config
    ax.grid(True, linestyle="dotted", alpha=1)
    ax.legend(loc=4)

    ax.set_title(
        f"{model_name}, {classifier}, set {split}, fold {fold_id+1} set ROC Curve"
    )

    # X and Y limits
    # axes[0].set_xlim(xlim)
    # axes[0].set_ylim(ylim)

    plt.legend(loc="lower right")
    plt.tight_layout()
    final_savedir = os.path.join(
        savedir, f"{model_name}_{classifier}_{split}_{fold_id+1}_roc.png"
    )
    fig.savefig(final_savedir, bbox_inches="tight", format="png", dpi=300)
    # plt.show()

def plot_macro_roc_auc_curves(pkl, num_classes, output_path):
    plt.figure(figsize=(10, 10))

    mean_tpr_all = np.zeros((num_classes, 1001))
    mean_fpr_all = np.linspace(0, 1, 1001)

    for fold_id, fold_data in pkl.items():
        tpr_all = fold_data["test"]["tpr"]
        fpr_all = fold_data["test"]["fpr"]
        auc_all = fold_data["test"]["auc"]

        # Plot ROC curves for each class
        for i in range(num_classes):
            plt.plot(
                fpr_all[i],
                tpr_all[i],
                lw=1,
                alpha=0.3,
                label=f"Fold {fold_id + 1} Class {i + 1} (AUC={auc_all[i]:.2f})",
            )

            # Compute mean TPR for each class
            interp_tpr = np.interp(mean_fpr_all, fpr_all[i], tpr_all[i])
            interp_tpr[0] = 0.0
            mean_tpr_all[i] += interp_tpr

    # Compute macro average ROC curve
    mean_tpr_all /= len(pkl)
    mean_tpr_all[-1] = 1.0
    mean_auc_all = auc(mean_fpr_all, mean_tpr_all)

    # Plot macro average ROC curve
    plt.plot(
        mean_fpr_all,
        mean_tpr_all[-1],
        color="navy",
        label=f"Macro Average (AUC={mean_auc_all:.2f})",
        lw=2,
    )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="grey")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    plt.savefig(output_path + "test.png")
    plt.close()


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    m_names, c_names = get_paper_names()

    callers = [*range(1, 11)]
    num_classes = len(callers)
    save_macro_aucs(args.expdir, args.task, m_names, c_names)

    # classifiers = ["SVC", "RF", "LinearSVC", "AB"]
    # ds_type = "no-downsample"

    # for classifier in classifiers:
    #     plot_roc_curves(
    #         args.expdir, args.task, m_names, c_names, num_classes, ds_type, classifier
    #     )
    # read_distance_pkls(args.expdir, args.task, m_names, c_names)

    # for split in ["train", "val", "test"]:
    #     for fold_id in range(5):
    #         plot_roc_auc_curves(
    #             pkl,
    #             fold_id,
    #             model_name,
    #             num_classes,
    #             "dump",
    #             model_name,
    #             split,
    #             classifier,
    #         )

    # for split in ["train", "val", "test"]:
    #     plot_macro_roc_auc_curves(pkl, num_classes, "dump")
