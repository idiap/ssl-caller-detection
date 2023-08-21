# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import interp
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    PredefinedSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle

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
    parser.add_argument(
        "-d",
        "--downsample",
        action=argparse.BooleanOptionalAction,
        help=""",
        Downsample the data to have equal samples per class.")
        """,
    )
    parser.add_argument(
        "-c",
        "--classifier",
        choices=["LinearSVC", "SVC", "RF", "AB"],
        help="""
        Classifier to classify the data points.
        """,
    )
    return parser


def create_grid(scalar, estimator):
    if scalar == "standard":
        scalar_method = StandardScaler()
    elif scalar == "robust":
        scalar_method = RobustScaler()

    GRID = [
        {
            "scaler": [scalar_method],
            "estimator__random_state": [RANDOM_SEED],
        }
    ]

    if "RF" in estimator:
        GRID[0]["estimator"] = [RandomForestClassifier()]
        GRID[0]["estimator__max_features"] = ["auto", "sqrt", "log2"]
        GRID[0]["estimator__criterion"] = ["gini", "entropy"]
        GRID[0]["estimator__min_samples_leaf"] = [1, 2, 4]
        GRID[0]["estimator__n_estimators"] = [50, 500, 1000, 2000]

    elif "AB" in estimator:
        GRID[0]["estimator"] = [AdaBoostClassifier()]
        GRID[0]["estimator__learning_rate"] = [0.1, 0.2, 0.5, 1]
        GRID[0]["estimator__algorithm"] = ["SAMME", "SAMME.R"]
        GRID[0]["estimator__n_estimators"] = [50, 500, 1000, 2000]

    elif "LinearSVC" in estimator:
        GRID[0]["estimator"] = [LinearSVC()]
        GRID[0]["estimator__loss"] = ["squared_hinge"]
        GRID[0]["estimator__C"] = np.logspace(0, -5, num=6)
        GRID[0]["estimator__class_weight"] = ["balanced", None]
        GRID[0]["estimator__max_iter"] = [10000]

    elif "SVC" in estimator:
        GRID[0]["estimator"] = [SVC(probability=True)]
        GRID[0]["estimator__C"] = np.logspace(0, -5, num=6)
        GRID[0]["estimator__kernel"] = ["rbf", "linear", "poly"]
        GRID[0]["estimator__gamma"] = ["scale", "auto"]
        GRID[0]["estimator__probability"] = [True]

    else:
        raise Exception(f"the estimator {estimator} is not defined.")

    return GRID


def make_dict_json_serializable(meta_dict: dict) -> dict:
    cleaned_meta_dict = meta_dict.copy()
    for key in cleaned_meta_dict:
        if type(cleaned_meta_dict[key]) not in [
            str,
            float,
            int,
        ]:  # , np.float64
            cleaned_meta_dict[key] = str(cleaned_meta_dict[key])
    return cleaned_meta_dict


def compute_eer(fpr, tpr):
    """Returns equal error rate (EER)."""
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer


def randomColorGenerator(number_of_colors=1, seed=0):
    """Generate list of random colors"""
    np.random.seed(seed)
    return [
        "#"
        + "".join(
            [np.random.choice(list("0123456789ABCDEF")) for j in range(6)]
        )
        for i in range(number_of_colors)
    ]


def plot_roc_auc_curves(
    fpr,
    tpr,
    auc_metric,
    num_classes,
    savedir,
    expname,
    group,
    classifier,
    seed=RANDOM_SEED,
    xlim=(-0.0025, 0.03),
    ylim=(0.99, 1.001),
):
    """Plot ROC AUC Curves"""
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

    # Baseline
    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    # Config
    ax.grid(True, linestyle="dotted", alpha=1)
    ax.legend(loc=4)

    # X and Y limits
    # axes[0].set_xlim(xlim)
    # axes[0].set_ylim(ylim)

    plt.legend(loc="lower right")
    plt.tight_layout()
    final_savedir = os.path.join(
        savedir, f"{expname}_{classifier}_{group}_roc.pdf"
    )
    fig.savefig(final_savedir, bbox_inches="tight", format="pdf", dpi=300)
    plt.show()


def plot_confusion_matrix(y, y_hat, target_names, savedir, expname):
    disp = ConfusionMatrixDisplay.from_predictions(
        y,
        y_hat,
        normalize="true",
        display_labels=target_names,
        include_values=True,
        values_format=".1g",
    )

    plt.savefig(
        os.path.join(savedir, expname + "_conf_matrix.pdf"),
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )


def save_cm(cm, target_names, savedir, expname, classifier):
    conf_matrix = pd.DataFrame(cm, index=target_names, columns=target_names)
    final_savedir = os.path.join(
        savedir, f"{expname}_{classifier}_conf_matrix.pkl"
    )
    utils.dump_features(conf_matrix, final_savedir)


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


def return_predictions_metrics(
    estimator,
    X,
    y,
    y_ohe,
    metrics,
    group,
    classification_method,
    target_names,
    savedir,
    expname,
    save_preds_csv=False,
):
    """
    Calculate and return all the metrics.
    """

    # Get predictions
    y_hat = estimator.predict(X)

    # Get scores
    accuracy = accuracy_score(y, y_hat)
    f1_score_macro = f1_score(y, y_hat, average="macro")
    f1_score_micro = f1_score(y, y_hat, average="micro")
    f1_score_weighted = f1_score(y, y_hat, average="weighted")
    uar = recall_score(y, y_hat, average="macro")
    cm = confusion_matrix(y, y_hat, labels=target_names, normalize="true")
    report = classification_report(
        y, y_hat, target_names=target_names, output_dict=True
    )

    # Get probabilities in funciton of the classifier
    if "LinearSVC" in classification_method:
        probs = estimator.decision_function(X)
        preds = probs  # only for binary, no ?
    elif "SVC" in classification_method:
        probs = estimator.predict_proba(X)
        preds = np.argmax(
            probs, axis=1
        )  # these go from 0-9, where as y_hat goes from 1-10
    else:
        probs = estimator.predict_proba(X)
        preds = probs[:, 1]

    # Get FPR, TPR, and thresholds
    fpr, tpr, thresholds, auc_metric = make_fpr_tpr_auc_dicts(
        y, y_ohe, probs, target_names
    )

    # Compute EER
    eer_micro = compute_eer(fpr["micro"], tpr["micro"])
    eer_macro = compute_eer(fpr["macro"], tpr["macro"])

    # Plot curves
    # plot_roc_auc_curves(
    #     fpr,
    #     tpr,
    #     auc_metric,
    #     len(target_names),
    #     savedir,
    #     expname,
    #     group,
    #     classification_method,
    # )

    # Save and plot CM
    # plot_confusion_matrix(y, y_hat, target_names, savedir, expname)

    print(f"--> AUC Macro: {np.round(auc_metric['macro'], 2)}")
    print(f"--> UAR: {np.round(uar, 2)}")
    print(f"--> F1 Macro: {np.round(f1_score_macro, 2)}")

    # Save metrics
    metrics[f"{group}"]["uar"] = uar
    metrics[f"{group}"]["accuracy"] = accuracy
    metrics[f"{group}"]["f1_score_macro"] = f1_score_macro
    metrics[f"{group}"]["f1_score_micro"] = f1_score_micro
    metrics[f"{group}"]["f1_score_weighted"] = f1_score_weighted
    metrics[f"{group}"]["cm"] = cm
    metrics[f"{group}"]["fpr"] = fpr
    metrics[f"{group}"]["tpr"] = tpr
    metrics[f"{group}"]["thresholds"] = thresholds
    metrics[f"{group}"]["auc"] = auc_metric
    metrics[f"{group}"]["eer_macro"] = eer_macro
    metrics[f"{group}"]["eer_micro"] = eer_micro
    metrics[f"{group}"]["report"] = report

    # Save predictions and probabilities
    if save_preds_csv:
        df_predictions = pd.DataFrame(
            {"preds": preds.tolist(), "class": y_hat.tolist()}
        )
        final_savedir = os.path.join(
            savedir,
            f"{expname}_{classification_method}_predictions_{group}.csv",
        )
        df_predictions.to_csv(final_savedir, index=False)

    return metrics


def macro_auc_score_func(X, clf_method, estimator):
    # Get probabilities in funciton of the classifier
    if "LinearSVC" in clf_method:
        probs = estimator.decision_function(X)
    elif "SVC" in clf_method:
        probs = estimator.predict_proba(X)
    else:
        probs = estimator.predict_proba(X)

    return roc_auc_score(y, probs, multi_class="ovr", average="macro")


def grid_search_fold(
    f, X_train, y_train, X_val, y_val, X_test, y_test, args, shuffled=False
):
    """
    Grid search for the best hyperparameters for all classifiers for the given fold.
    """

    # Shuffle data within a fold
    if shuffled:
        print("Shuffling the data...")
        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)
        X_test, y_test = shuffle(X_test, y_test)

    # Print shapes with counts
    counts = np.unique(y_train, return_counts=True)[1]
    print(f"Train: {X_train.shape}, {y_train.shape}", counts)

    counts = np.unique(y_val, return_counts=True)[1]
    print(f"Val: {X_val.shape}, {y_val.shape}", counts)

    counts = np.unique(y_test, return_counts=True)[1]
    print(f"Test: {X_test.shape}, {y_test.shape}", counts)

    # Variables
    num_train = X_train.shape[0]
    num_val = X_val.shape[0]
    split_indices = np.repeat([-1, 0], [num_train, num_val])
    split = PredefinedSplit(split_indices)

    # One-Hot Encode labels
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_val_ohe = lb.transform(y_val)
    y_test_ohe = lb.transform(y_test)

    # Join train and validation sets for CV
    X = np.append(X_train, X_val, axis=0)
    y = np.append(y_train, y_val, axis=0)

    # GridSearch variables
    GRID = create_grid(scalar="standard", estimator=args.classifier)
    PIPELINE = Pipeline(
        [("scaler", StandardScaler()), ("estimator", RandomForestClassifier())]
    )
    # macro_auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', average='macro')

    # Instantiate Grid Search -- this is not doing CV but using a predefined splits which are constant
    grid_search = GridSearchCV(
        estimator=PIPELINE,
        param_grid=GRID,
        scoring="f1_macro",  # or: "f1_macro"
        n_jobs=-1,  # use all cpus
        cv=split,  # make stratified ?
        refit=True,
        verbose=1,
        return_train_score=False,
    )

    # Search for best parameter using train and val sets using CV
    print(
        f"Computing optimal parameters for split {f+1} using a GridSearch ..."
    )
    grid_search.fit(X, y)

    # Clone best estimator
    best_estimator = grid_search.best_estimator_
    estimator = clone(best_estimator, safe=False)

    # Save classifier
    print(f"Saving the optimal estimator for fold {f+1}...")
    final_savedir = os.path.join(
        savedir, f"{args.expname}_{args.classifier}_fold{f+1}_clf.pkl"
    )
    utils.dump_features(estimator, final_savedir)

    # Metrics
    metrics = {g: {} for g in groups}
    metrics["params"] = make_dict_json_serializable(grid_search.best_params_)

    # Save grid search CV results
    print("Saving grid search results ...")
    final_savedir = os.path.join(
        savedir, f"{args.expname}_{args.classifier}_fold{f+1}_grid_search.csv"
    )
    pd.DataFrame(grid_search.cv_results_).to_csv(final_savedir, index=False)

    # Fit clone of best estimator again on train for val predictions
    estimator.fit(X_train, y_train)

    # Shouldn't we do fit_transform for the other sets ?

    ssets = {
        "train": {"X": X_train, "y": y_train, "y_ohe": y_train_ohe},
        "val": {"X": X_val, "y": y_val, "y_ohe": y_val_ohe},
        "test": {"X": X_test, "y": y_test, "y_ohe": y_test_ohe},
    }

    # Iterate over all groups and get metrics
    for group in groups:
        print(f"Computing {group} metrics ...")

        # Chose estimator
        selected_estimator = best_estimator if group in ["test"] else estimator

        metrics = return_predictions_metrics(
            estimator=selected_estimator,
            X=ssets[group]["X"],
            y=ssets[group]["y"],
            y_ohe=ssets[group]["y_ohe"],
            metrics=metrics,
            group=group,
            classification_method=args.classifier,
            target_names=target_names,
            savedir=savedir,
            expname=args.expname,
        )

    # Return metrics for the given fold
    return metrics


parser = get_argument_parser()
args = parser.parse_args()

if __name__ == "__main__":
    # Task
    task = args.task
    groups = ["train", "val", "test"]
    random.seed(RANDOM_SEED)

    # Save path
    ds_path = "downsample" if args.downsample else "no-downsample"
    savedir = os.path.join(
        args.savedir, ds_path, args.expname, args.task, args.classifier
    )
    os.makedirs(savedir, exist_ok=True)

    # Load the mapping between uids and pickleids
    uids_to_pickleid = utils.load_features(
        "marmoset_lists/uids_to_pickleid.pkl"
    )
    pickleid_to_uids = utils.load_features(
        "marmoset_lists/pickleid_to_uids.pkl"
    )

    # Load the mapping between calltypes and indices
    calltype_to_index_path = "dataset/calltype_to_index_reduced.json"

    # Load the callers
    callers = [*range(1, 11)]

    # Labels
    if task == "calltypeID":
        with open(calltype_to_index_path) as f:
            target_names = json.load(f)
    else:
        target_names = callers

    # Read functionals
    print("Reading functionals...")
    funcs_dict, labels = utils.load_functionals_covar(
        args.expdir, ds_path, groups, args.expname, task, callers
    )

    # Make func_dict into a single numpy array
    X_train = np.concatenate([funcs_dict["train"][c] for c in callers], axis=0)
    y_train = np.concatenate([labels["train"][c] for c in callers], axis=0)

    X_val = np.concatenate([funcs_dict["val"][c] for c in callers], axis=0)
    y_val = np.concatenate([labels["val"][c] for c in callers], axis=0)

    X_test = np.concatenate([funcs_dict["test"][c] for c in callers], axis=0)
    y_test = np.concatenate([labels["test"][c] for c in callers], axis=0)

    # Concantenate into a single X and Y
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)

    num_folds = 5  # num_splits
    metrics_all_folds = {}

    # Make 10 folds, and for each fold do a grid search hyper param optimization
    print(f"Splitting data into {num_folds} folds...")
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED)

    # Iterate over the folds
    for f, (train_index, test_index) in enumerate(kf.split(X)):
        print("=======================================================")
        print(f"Split {f+1} of {num_folds}")

        # Split into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Split into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
        )

        metrics_fold = grid_search_fold(
            f, X_train, y_train, X_val, y_val, X_test, y_test, args
        )

        metrics_all_folds[f] = metrics_fold

    # Save metrics for all folds
    print("Saving computed metrics for all folds ...")
    final_savedir = os.path.join(
        savedir, f"{args.expname}_{args.classifier}_all_folds_metrics.pkl"
    )
    utils.dump_features(metrics_all_folds, final_savedir)

    print("Done!")
