__author__ = 'teemu kanstren'

import pandas as pd
import test_predictor

#check if given parameter can be interpreted as a numerical value
def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

#convert given set of paramaters to integer values
#this at least cuts the excess float decimals if they are there
def convert_int_params(names, params):
    for int_type in names:
        #sometimes the parameters can be choices between options or numerical values. like "log2" vs "1-10"
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params

#convert float parameters to 3 digit precision strings
#just for simpler diplay and all
def convert_float_params(names, params):
    for float_type in names:
        raw_val = params[float_type]
        if is_number(raw_val):
            params[float_type] = '{:.3f}'.format(raw_val)
    return params

def create_misclassified_dataframe(result, y):
    oof_series = pd.Series(result.oof_predictions[result.misclassified_indices])
    oof_series.index = y[result.misclassified_indices].index
    miss_scale_raw = y[result.misclassified_indices] - result.oof_predictions[result.misclassified_indices]
    miss_scale_abs = abs(miss_scale_raw)
    df_miss_scale = pd.concat([miss_scale_raw, miss_scale_abs, oof_series, y[result.misclassified_indices]], axis=1)
    df_miss_scale.columns = ["Raw_Diff", "Abs_Diff", "Prediction", "Actual"]
    result.df_misses = df_miss_scale

class OptimizerResult:
    avg_accuracy = None,
    misclassified_indices = None,
    misclassified_expected = None,
    misclassified_actual = None,
    oof_predictions = None,
    predictions = None,
    df_misses = None,
    all_accuracies = None,
    all_losses = None,
    all_params = None,
    best_params = None,
    #the final classifier created from optimized hyperparameters, to allow feature weight access etc
    clf = None,


import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import catboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import hyperopt
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


def hyperopt_run_search(parent):
    space = parent.create_hyperspace()
    trials = Trials()
    best = fmin(fn=parent.objective_sklearn,
                space=space,
                algo=tpe.suggest,
                max_evals=parent.n_trials,
               trials=trials)

    # find the trial with lowest loss value. this is what we consider the best one
    idx = np.argmin(trials.losses())
    print(idx)

    print(trials.trials[idx])

    params = trials.trials[idx]["result"]["params"]
    print(params)
    return params, idx

def hyperopt_search_classify(parent, X_cols, df_train, df_test, y_param, train_pct, stratify_train):
    parent.y = y_param
    parent.X = df_train[X_cols]
    parent.X_test = df_test[X_cols]
    #convert train_pct into percentage to drop, i.e. the test set from split

    #resetting index since sliced dataframes can cause issues in stratified split indexing later
    #this might be something to consider in special cases
    parent.X = parent.X.reset_index(drop=True)
    parent.X_test = parent.X_test.reset_index(drop=True)
    parent.y = parent.y.reset_index(drop=True)

    if train_pct is not None:
        test_pct = 1 - train_pct
        train_indices, test_indices = train_test_split(parent.X.index, test_size = test_pct, stratify=stratify_train)
        parent.train_indices = train_indices


    # use 2 classes as this is a binary classification
    # the second param is the number of rows to use for training
    params, idx = hyperopt_run_search(parent)
    print(params)

    clf = parent.classifier(**params)

    search_results = test_predictor.stratified_test_prediction_avg_vote(parent, clf, parent.X, parent.X_test, parent.y, n_folds=parent.n_folds,
                                                         n_classes=parent.n_classes, fit_params=parent.fit_params,
                                                        train_indices=parent.train_indices, use_calibration=parent.use_calibration)
    search_results.all_accuracies = parent.all_accuracies
    search_results.all_losses = parent.all_losses
    search_results.all_params = parent.all_params
    search_results.best_params = params
    search_results.clf = clf
    return search_results

def hyperopt_objective_run(parent, params):
    score, logloss = fit_cv(parent, parent.X, parent.y, params, parent.fit_params, parent.n_classes, parent.classifier,
                        parent.use_calibration, parent.n_folds, parent.print_summary, verbosity=parent.verbosity,
                       train_indices=parent.train_indices)
    parent.all_params.append(params)
    parent.all_accuracies.append(score)
    parent.all_losses.append(logloss)
    if parent.verbosity == 0:
        if parent.print_summary:
            print("Score {:.3f}".format(score))
    else:
        print("Score {:.3f} params {}".format(score, params))
    #using logloss here for the loss but uncommenting line below calculates it from average accuracy
#    loss = 1 - score
    loss = logloss
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result