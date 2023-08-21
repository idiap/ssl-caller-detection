# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>
#
# Code for the paper:
# "Can Self-Supervised Neural Representations Pre-Trained 
# on Human Speech distinguish Animal Callers?"
# Authors: Eklavya Sarkar, Mathew Magimai Doss

import json
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import interp
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
import utils

RANDOM_SEED = 42
PIPELINE = Pipeline(
    [("scaler", StandardScaler()), ("estimator", RandomForestClassifier())]
)

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
        if type(cleaned_meta_dict[key]) not in [str, float, int]:  # , np.float64
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
        "#" + "".join([np.random.choice(list("0123456789ABCDEF")) for j in range(6)])
        for i in range(number_of_colors)
    ]


def plot_roc_auc_curves(
    fpr,
    tpr,
    auc_metric,
    num_classes,
    xlim=(-0.0025, 0.03),
    ylim=(0.99, 1.001),
    seed=RANDOM_SEED,
    save_title=None,
):
    """Plot ROC AUC Curves"""
    fig, axes = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(10, 5))

    lw = 2
    axes[0].set_xlabel("False Positive Rate")
    axes[1].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    class_colors = randomColorGenerator(num_classes, seed)

    # Plot ROC curve for each class
    for i in range(num_classes):
        axes[0].plot(
            fpr[i],
            tpr[i],
            lw=lw,
            color=class_colors[i],
            label="{0} ({1:0.2f}%)" "".format(str(i + 1), auc_metric[i] * 100),
        )

    # Plot micro and macro curves
    axes[0].plot(
        fpr["micro"], 
        tpr["micro"], 
        lw=lw,
        label="Micro avg ({:0.2f}%)" "".format(auc_metric["micro"] * 100),
        linestyle=":",
        color="deeppink",
    )
    axes[0].plot(
        fpr["macro"],
        tpr["macro"],
        lw=lw,
        label="Macro avg ({:0.2f}%)" "".format(auc_metric["macro"] * 100),
        linestyle=":",
        color="navy",
    )

    # Baseline
    axes[0].plot([0, 1], [0, 1], color="k", linestyle="--")

    # Config
    axes[0].grid(True, linestyle="dotted", alpha=1)
    axes[0].legend(loc=4)

    # X and Y limits
    # axes[0].set_xlim(xlim)
    # axes[0].set_ylim(ylim)

    plt.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{save_title}.pdf", bbox_inches="tight", format="pdf", dpi=300)
    plt.show()


def make_fpr_tpr_auc_dicts(y_ohe, probs_list, class_names):
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
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_ohe[:, i], probs_list[:, i], pos_label=1)
        auc_metric[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area.
    fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_ohe.ravel(), probs_list.ravel())
    auc_metric["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # 1. First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # 2. Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # 3. Finally average it and compute AUC
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc_metric["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, thresholds, auc_metric

def save_cm(cm, target_names, save_path, exp_name):
    conf_matrix = pd.DataFrame(cm, index=target_names, columns=target_names)
    final_save_path = os.path.join(save_path, exp_name + "_conf_matrix.pkl")
    utils.dump_features(conf_matrix, final_save_path)


def plot_confusion_matrix(y, y_hat, target_names, save_path, exp_name):
    disp = ConfusionMatrixDisplay.from_predictions(
        y,
        y_hat,
        normalize="true",
        display_labels=target_names,
        include_values=True,
        values_format=".1g",
    )

    plt.savefig(
        os.path.join(save_path, exp_name + "_conf_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )

def return_predictions_metrics(estimator, 
                               X, 
                               y, 
                               y_ohe,
                               metrics, 
                               group, 
                               classification_method,
                               target_names,
                               eval_dir,
                               save_preds_csv=False,
                               save_path,
                               exp_name
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
    report = classification_report(y, y_hat, target_names=target_names, output_dict=True)

    # Get probabilities in funciton of the classifier
    if "LinearSVC" in classification_method:
        probs = estimator.decision_function(X)
        preds = probs  # only for binary, no ?
    elif "SVC" in classification_method:
        probs = estimator.predict_proba(X)
        preds = np.argmax(probs, axis=1) # these go from 0-9, where as y_hat goes from 1-10
    else:
        probs = estimator.predict_proba(X)
        preds = probs[:, 1]

    # Get FPR, TPR, and thresholds
    fpr, tpr, thresholds, auc_metric = make_fpr_tpr_auc_dicts(y_ohe, probs, target_names)

    # Compute EER
    eer_micro = compute_eer(fpr["micro"], tpr["micro"])
    eer_macro = compute_eer(fpr["macro"], tpr["macro"])

    # Plot curves
    plot_roc_auc_curves(fpr, tpr, auc_metric, len(target_names), save_title=f"{group}_roc_auc")

    # Save and plot CM
    save_cm(cm, target_names, save_path, exp_name)
    plot_confusion_matrix(y, y_hat, target_names, save_path, exp_name)

    print(
        f"Group:" {group},
        f"AUC: {auc_metric}",
        f"UAR: {uar}\n",
        pd.DataFrame.from_dict(report).T.round(2)
    )

    # Save metrics
    metrics[f"{group}"]["uar"] = uar
    metrics[f"{group}"]["accuracy"] = accuracy
    metrics[f"{group}"]["f1_score_macro"] = f1_score_macro
    metrics[f"{group}"]["f1_score_micro"] = f1_score_micro
    metrics[f"{group}"]["f1_score_weighted"] = f1_score_weighted
    metrics[f"{group}"]["cm"] = cm.tolist()
    metrics[f"{group}"]["fpr"] = fpr
    metrics[f"{group}"]["tpr"] = tpr
    metrics[f"{group}"]["thresholds"] = thresholds
    metrics[f"{group}"]["auc"] = auc_metric
    metrics[f"{group}"]["eer_macro"] = eer_macro
    metrics[f"{group}"]["eer_micro"] = eer_micro
    # metrics[f"{group}"]["eer_th"] = eer_th
    metrics[f"{group}"]["report"] = report

    # Save predictions and probabilities
    if save_preds_csv:
        df_predictions = pd.DataFrame({"preds": preds.tolist(), "class": y_hat.tolist()})
        df_predictions.to_csv(os.path.join(eval_dir, f"predictions_{group}.csv"), index=False)

    return metrics


def run_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    eval_dir,
    classification_method,
    scalar,
    target_names,
    save_path,
    exp_name
):

    # Fix seed
    random.seed(RANDOM_SEED)

    # Create grid search's GRID
    GRID = create_grid(scalar=scalar, estimator=classification_method)

    # Variables
    num_train = X_train.shape[0]
    num_val = X_val.shape[0]
    split_indices = np.repeat([-1, 0], [num_train, num_val])
    split = PredefinedSplit(split_indices)
    groups = ["train", "val", "test"]

    # One-Hot Encode labels
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_val_ohe = lb.transform(y_val)
    y_test_ohe = lb.transform(y_test)

    print(f"y_ohe.shape: {y_train_ohe.shape}")

    # Join train and validation sets for CV
    X = np.append(X_train, X_val, axis=0)
    y = np.append(y_train, y_val, axis=0)

    # Instantiate Grid Search
    grid_search = GridSearchCV(
        estimator=PIPELINE,
        param_grid=GRID,
        scoring="f1_macro",  # define macro_roc_auc ideall
        n_jobs=-1,  # use all cpus
        cv=split,
        refit=True,
        verbose=1,
        return_train_score=False,
    )

    # Search for best parameter using train and val sets using CV
    print("Computing optimal parameters using a CV GridSearch ...")
    grid_search.fit(X, y)

    # Clone best estimator
    best_estimator = grid_search.best_estimator_
    estimator = clone(best_estimator, safe=False)
    
    # Save classifier
    print("Saving the optimal estimator...") 
    final_save_path = os.path.join(save_path, exp_name + "_clf.pkl")
    utils.dump_features(estimator, final_save_path)

    # Metrics
    metrics = {g: {} for g in groups}
    metrics["params"] = make_dict_json_serializable(grid_search.best_params_)

    # Save grid search CV results
    print("Saving grid search results ...")
    pd.DataFrame(grid_search.cv_results_).to_csv(
        os.path.join(eval_dir, f"grid_search.csv"), index=False
    )

    # Fit clone of best estimator again on train for val predictions
    estimator.fit(X_train, y_train)

    # Iterate over all groups and get metrics
    for group in groups:
        print("Computing {group} metrics ...")
        metrics = return_predictions_metrics(
            estimator=estimator,
            X=f"{X}_{group}",
            y=f"{y}_{group}",
            y_ohe=f"{y}_{group}_ohe",
            metrics=metrics,
            group=group,
            classification_method=classification_method,
            target_names=target_names,
            eval_dir=eval_dir,
            save_path=save_path,
            exp_name=exp_name
        )

    # Save metrics
    print("Saving computed metrics ...")
    final_save_path = os.path.join(save_path, exp_name + "_metrics.pkl")
    utils.dump_features(metrics, final_save_path)
