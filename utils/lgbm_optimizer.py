__author__ = 'teemu kanstren'

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
import hyperopt
from cross_validator import fit_cv
import test_predictor

def create_classifier(params):
    clf = lgbm.LGBMClassifier(
        **params
    )
    return clf

def objective_sklearn(params):
    params['num_leaves'] = int(params['num_leaves'])
    params['min_child_samples'] = int(params['min_child_samples'])
    params['subsample_for_bin'] = int(params['subsample_for_bin'])

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
    score = fit_cv(X, y, params, fit_params, create_classifier)
    print("Score {:.3f} params {}".format(score, params))
    loss = 1 - score
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result

def optimize_lgbm():
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
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

    clf = lgbm.LGBMClassifier(
        **params
    #    **best
    )

    search_results = test_predictor.stratified_test_prediction_avg_vote(clf, df_train_sum, df_test_sum, y)
    predictions = search_results["predictions"]

    ss = pd.read_csv('../input/career-con-2019/sample_submission.csv')
    ss['surface'] = encoder.inverse_transform(sub_preds.argmax(axis=1))
    ss.to_csv('lgbm.csv', index=False)
    ss.head(10)

