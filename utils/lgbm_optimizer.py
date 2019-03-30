__author__ = 'teemu kanstren'

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import hyperopt
import test_predictor
import opt_utils

def fit_cv(X, y, params, fit_params):
    score = 0
    n_folds = 5
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        print('-' * 20, i, '-' * 20)

        clf = lgbm.LGBMClassifier(**params)
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        # verbose = print loss at every "verbose" rounds
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=1000, **fit_params)
        oof_preds = np.zeros((X.shape[0]))
        oof_preds[test_index] = clf.predict(X.iloc[test_index])
        score += clf.score(X.iloc[test_index], y[test_index])
        print('score ', clf.score(X.iloc[test_index], y[test_index]))
        importances = clf.feature_importances_
        features = X.columns
    return score / n_folds


def objective_sklearn(params):
    int_types = ["num_leaves", "min_child_samples", "subsample_for_bin", "min_data_in_leaf"]
    params = opt_utils.convert_int_params(int_types, params)

    params['colsample_bytree'] = '{:.3f}'.format(params['colsample_bytree'])
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    #    print("running with params:"+str(params))

    using_dart = params['boosting_type'] == "dart"

    fit_params = {"eval_metric": "multi_logloss"}
    if using_dart:
        n_estimators = 500
    else:
        n_estimators = 10000
        fit_params["early_stopping_rounds"] = 100
    params["n_estimators"] = n_estimators
    score = fit_cv(X, y, params, fit_params)
    print("Score {:.3f} params {}".format(score, params))
    loss = 1 - score
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result

def optimize_lgbm():
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
    #https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
    space = {
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'boosting_type': hp.choice('boosting_type',
                                   [{'boosting_type': 'gbdt',
                                     'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                    {'boosting_type': 'dart',
                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                    {'boosting_type': 'goss'}]),
        'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        #what is negative value for?
        #'learning_rate': hp.loguniform('learning_rate', -7, 0),
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
        'objective': "multiclass", "num_class": 9
    }
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

    params = optimize_lgbm()
    print(params)

    clf = lgbm.LGBMClassifier(**params)

    search_results = test_predictor.stratified_test_prediction_avg_vote(clf, df_train_sum, df_test_sum, y)
    predictions = search_results["predictions"]

    ss = pd.read_csv('../input/career-con-2019/sample_submission.csv')
    ss['surface'] = encoder.inverse_transform(predictions.argmax(axis=1))
    ss.to_csv('lgbm.csv', index=False)
    ss.head(10)

