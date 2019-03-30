__author__ = 'teemu kanstren'

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import hyperopt
import test_predictor
import opt_utils

def fit_cv(X, y, params, fit_params):
    print("params:"+str(params))
    score = 0
    n_folds = 5
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        print('-' * 20, i, '-' * 20)

        clf = RandomForestClassifier(**params)
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        clf.fit(X_train, y_train, **fit_params)
        oof_preds = np.zeros((X.shape[0]))
        oof_preds[test_index] = clf.predict(X.iloc[test_index])
        score += clf.score(X.iloc[test_index], y[test_index])
        print('score ', clf.score(X.iloc[test_index], y[test_index]))
        importances = clf.feature_importances_
        features = X.columns
    return score/n_folds

def objective_sklearn(params):
    int_types = ["n_estimators", "min_samples_leaf", "min_samples_split", "max_features"]
    params = opt_utils.convert_int_params(int_types, params)
    score = fit_cv(X, y, params, {})
    print("Score {:.3f} params {}".format(score, params))
    loss = 1 - score
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result


def optimize_rf():
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    space = {
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        # 'scale': hp.choice('scale', [0, 1]),
        # 'normalize': hp.choice('normalize', [0, 1]),
        'bootstrap': hp.choice('bootstrap', [True, False]),
        # nested choice: https://medium.com/vooban-ai/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters-e3102814b919
        'max_depth': hp.choice('max_depth', [None, hp.quniform('max_depth_num', 10, 100, 10)]),
        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None, hp.quniform('max_features_num', 1, 5, 1)]),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 4),
        'class_weight': hp.choice('class_weight', ["balanced", None]),
        'n_estimators': hp.quniform('n_estimators', 200, 2000, 200),
        'n_jobs': -1
    }
    # save and reload trials for hyperopt state: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/hyperopt_optimize.py

    trials = Trials()
    best = fmin(fn=objective_sklearn,
                space=space,
                algo=tpe.suggest,
                max_evals=2,
               trials=trials)

    idx = np.argmin(trials.losses())
    print(idx)

    print(trials.trials[idx])

    params = trials.trials[idx]["result"]["params"]
    print(params)
    return params


if __name__== "__main__":
    df_train_sum = pd.read_csv("./features_train_scaled_sum.csv")
    df_test_sum = pd.read_csv("./features_test_scaled_sum.csv")
    df_y = pd.read_csv("./y_train.csv")
    cols_to_drop = ["series_id"]
    df_train_sum.drop(cols_to_drop, axis=1, inplace=True)
    df_test_sum.drop(cols_to_drop, axis=1, inplace=True)

    # encode class values as integers so they work as targets for the prediction algorithm
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_y["surface"])
    y_count = len(list(encoder.classes_))
    cols_to_drop = ["series_id"]
    #df_train_sum.drop(cols_to_drop, axis=1, inplace=True)
    #df_test_sum.drop(cols_to_drop, axis=1, inplace=True)
    X = df_train_sum

    params = optimize_rf()
    print(params)

    clf = RandomForestClassifier(**params)

    search_results = test_predictor.stratified_test_prediction_avg_vote(clf, df_train_sum, df_test_sum, y)
    predictions = search_results["predictions"]

    ss = pd.read_csv('./career-con-2019/sample_submission.csv')
    ss['surface'] = encoder.inverse_transform(predictions.argmax(axis=1))
    ss.to_csv('rf.csv', index=False)
    ss.head(10)