__author__ = 'teemu kanstren'

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

# run n_folds of cross validation on the data
# averages fold results
def fit_cv(X, y, params, fit_params, n_classes, classifier, max_n, n_folds, print_summary, verbosity):
    # cut the data if max_n is set
    if max_n is not None:
        X = X[:max_n]
        y = y[:max_n]

    fit_params = fit_params.copy()
    use_eval = fit_params.pop("use_eval_set")
    score = 0
    acc_score = 0
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    if print_summary:
        print(f"Running {n_folds} folds...")
    oof_preds = np.zeros((X.shape[0], n_classes))
    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        if verbosity > 0:
            print('-' * 20, f"RUNNING FOLD: {i}/{n_folds}", '-' * 20)

        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        #https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
        #clf = catboost.CatBoostClassifier(**params)
        clf = classifier(**params)
        # verbose = print loss at every "verbose" rounds.
        #if 100 it prints progress 100,200,300,... iterations
        if use_eval:
            clf.fit(X_train, y_train, eval_set=(X_test, y_test), **fit_params)
        else:
            clf.fit(X_train, y_train, **fit_params)
        oof_preds[test_index] = clf.predict_proba(X.iloc[test_index])
        #score += clf.score(X.iloc[test_index], y[test_index])
        acc_score += accuracy_score(y[test_index], oof_preds[test_index][:,1] >= 0.5)
        # print('score ', clf.score(X.iloc[test_index], y[test_index]))
        #importances = clf.feature_importances_
        features = X.columns
    #accuracy is calculated each fold so divide by n_folds.
    #not n_folds -1 because it is not sum by row but overall sum of accuracy of all test indices
    total_acc_score = acc_score / n_folds
    logloss = log_loss(y, oof_preds)
    if print_summary:
        print(f"total acc: {total_acc_score}, logloss={logloss}")
    return total_acc_score, logloss
