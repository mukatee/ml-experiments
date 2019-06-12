__author__ = 'teemu kanstren'

from test_predictor import *
from opt_utils import *

import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import hyperopt

# how many CV folds to do on the data
n_folds = 5
# max number of rows to use for X and y. to reduce time and compare options faster
max_n = None
# max number of trials hyperopt runs
n_trials = 1000
verbosity = 0
from sklearn.metrics import accuracy_score, log_loss

all_accuracies = []
all_losses = []
all_params = []


# run n_folds of cross validation on the data
# averages fold results
def fit_cv(X, y, params, fit_params, n_classes):
    # cut the data if max_n is set
    if max_n is not None:
        X = X[:max_n]
        y = y[:max_n]

    score = 0
    acc_score = 0
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    if verbosity == 0:
        print(f"Running {n_folds} folds...")
    oof_preds = np.zeros((X.shape[0], n_classes))
    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        if verbosity > 0:
            print('-' * 20, f"RUNNING FOLD: {i}/{n_folds}", '-' * 20)

        clf = lgbm.LGBMClassifier(**params)
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        # verbose = print loss at every "verbose" rounds.
        #if 100 it prints progress 100,200,300,... iterations
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=verbosity, **fit_params)
        oof_preds[test_index] = clf.predict_proba(X.iloc[test_index])
        #score += clf.score(X.iloc[test_index], y[test_index])
        acc_score += accuracy_score(y[test_index], oof_preds[test_index][:,1] >= 0.5)
        # print('score ', clf.score(X.iloc[test_index], y[test_index]))
        importances = clf.feature_importances_
        features = X.columns
    #accuracy is calculated each fold so divide by n_folds.
    #not n_folds -1 because it is not sum by row but overall sum of accuracy of all test indices
    total_acc_score = acc_score / n_folds
    all_accuracies.append(total_acc_score)
    logloss = log_loss(y, oof_preds)
    all_losses.append(logloss)
    all_params.append(params)
    print(f"total acs: {total_acc_score}, logloss={logloss}")
    return total_acc_score, logloss

def create_fit_params(params):
    using_dart = params['boosting_type'] == "dart"
    if params["objective"] == "binary":
        # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        fit_params = {"eval_metric": ["binary_logloss", "auc"]}
    else:
        fit_params = {"eval_metric": "multi_logloss"}
    if using_dart:
        n_estimators = 2000
    else:
        n_estimators = 15000
        fit_params["early_stopping_rounds"] = 100
    params["n_estimators"] = n_estimators
    return fit_params


# this is the objective function the hyperopt aims to minimize
# i call it objective_sklearn because the lgbm functions called use sklearn API
def objective_sklearn(params):
    int_types = ["num_leaves", "min_child_samples", "subsample_for_bin", "min_data_in_leaf"]
    params = convert_int_params(int_types, params)

    params['colsample_bytree'] = '{:.3f}'.format(params['colsample_bytree'])
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    #    print("running with params:"+str(params))

    fit_params = create_fit_params(params)
    if params['objective'] == "binary":
        n_classes = 2
    else:
        n_classes = params["num_class"]

    score, logloss = fit_cv(X, y, params, fit_params, n_classes)
    if verbosity == 0:
        print("Score {:.3f}".format(score))
    else:
        print("Score {:.3f} params {}".format(score, params))
#    loss = 1 - score
    loss = logloss
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result


def optimize_lgbm(n_classes, max_n_search=None):
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
    # https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
    space = {
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'boosting_type': hp.choice('boosting_type',
                                   [{'boosting_type': 'gbdt',
                                     'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                    {'boosting_type': 'dart',
                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                    {'boosting_type': 'goss'}]),
        'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    }
    if n_classes > 2:
        space['objective'] = "multiclass"
        space["num_class"] = n_classes
    else:
        space['objective'] = "binary"
        #space["num_class"] = 1

    global max_n
    max_n = max_n_search
    trials = Trials()
    best = fmin(fn=objective_sklearn,
                space=space,
                algo=tpe.suggest,
                max_evals=n_trials,
                trials=trials)

    # find the trial with lowest loss value. this is what we consider the best one
    idx = np.argmin(trials.losses())
    print(idx)

    print(trials.trials[idx])

    # these should be the training parameters to use to achieve the best score in best trial
    params = trials.trials[idx]["result"]["params"]
    max_n = None

    print(params)
    return params


# run a search for exclusive multi-class classification
def classify_multiclass():
    df_train_sum = pd.read_csv("./test_data/kaggle_careercon_2019/features_train_scaled_sum.csv", nrows=100)
    df_test_sum = pd.read_csv("./test_data/kaggle_careercon_2019/features_test_scaled_sum.csv", nrows=100)
    df_y = pd.read_csv("./test_data/kaggle_careercon_2019/y_train.csv", nrows=100)
    cols_to_drop = ["series_id"]
    df_train_sum.drop(cols_to_drop, axis=1, inplace=True)
    df_test_sum.drop(cols_to_drop, axis=1, inplace=True)

    # encode class values as integers so they work as targets for the prediction algorithm
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_y["surface"])
    y_count = len(list(encoder.classes_))
    cols_to_drop = ["series_id"]
    # df_train_sum.drop(cols_to_drop, axis=1, inplace=True)
    # df_test_sum.drop(cols_to_drop, axis=1, inplace=True)
    X = df_train_sum

    params = optimize_lgbm(9)
    print(params)

    clf = lgbm.LGBMClassifier(**params)

    search_results = stratified_test_prediction_avg_vote(clf, df_train_sum, df_test_sum, y, use_eval_set=True, n_folds=n_folds)
    predictions = search_results["predictions"]

    ss = pd.read_csv('../input/sample_submission.csv')
    ss['surface'] = encoder.inverse_transform(predictions.argmax(axis=1))
    ss.to_csv('lgbm.csv', index=False)
    ss.head(10)


# run a search for binary classification
def classify_binary(X_cols, df_train, df_test, y_param):
    global X
    global y
    y = y_param
    nrows = max_n

    X = df_train[X_cols]
    X_test = df_test[X_cols]

    # use 2 classes as this is a binary classification
    # the second param is the number of rows to use for training
    params = optimize_lgbm(2, 5000)
    print(params)

    clf = lgbm.LGBMClassifier(**params)

    fit_params = create_fit_params(params)

    search_results = stratified_test_prediction_avg_vote(clf, X, X_test, y, use_eval_set=True,
                                                         n_folds=n_folds, n_classes=2, fit_params=fit_params, verbosity=verbosity)
    predictions = search_results["predictions"]
    oof_predictions = search_results["oof_predictions"]
    avg_accuracy = search_results["avg_accuracy"]
    misclassified_indices = search_results["misclassified_indices"]
    misclassified_samples_expected = search_results["misclassified_samples_expected"]
    misclassified_samples_actual = search_results["misclassified_samples_actual"]

    return predictions, oof_predictions, avg_accuracy, misclassified_indices

