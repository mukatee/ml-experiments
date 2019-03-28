__author__ = 'teemu kanstren'

import lightgbm as lgbm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def fit_cv(X, y, params, fit_params, create_classifier):
    score = 0
    n_folds = 5
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        print('-' * 20, i, '-' * 20)

        clf = create_classifier(params)
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        #verbose = print loss at every "verbose" rounds
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=1000, **fit_params)
        oof_preds = np.zeros((X.shape[0]))
        oof_preds[test_index] = clf.predict(X.iloc[test_index])
        score += clf.score(X.iloc[test_index], y[test_index])
        print('score ', clf.score(X.iloc[test_index], y[test_index]))
        importances = clf.feature_importances_
        features = X.columns
    return score/n_folds

