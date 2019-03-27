__author__ = 'teemu kanstren'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score, StratifiedKFold

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

    if using_dart:
        n_estimators = 500
        fit_params = {"eval_metric": "multi_logloss"}
    else:
        n_estimators = 10000,
        fit_params = {"early_stopping_rounds": 100, "eval_metric": "multi_logloss"}
    clf = lgbm.LGBMClassifier(
        n_estimators=500,
        **params
    )
    score = cross_val_score(clf, X, y, scoring="accuracy", fit_params=fit_params, cv=StratifiedKFold(n_splits=3)).mean()
    print("Score {:.3f} params {}".format(score, params))
    return score


def objective(params):
    global ITERATION
    ITERATION += 1
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    params['colsample_bytree'] = '{:.3f}'.format(params['colsample_bytree'])
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    #    print("running with params:"+str(params))

    train_set = lgbm.Dataset(X, label=y)
    start = timer()
    r = lgbm.cv(params, train_set, num_boost_round=10000, nfold=10, metrics='multi_logloss',
                early_stopping_rounds=100, verbose_eval=False, seed=50)
    run_time = timer() - start
    print("run time:" + str(run_time))
    # Highest score
    best_score = np.max(r['multi_logloss-mean'])
    # Loss must be minimized
    loss = 1 - best_score

    result = {'loss': loss, 'params': params, 'iteration': ITERATION,
              'estimators': n_estimators,
              'train_time': run_time, 'status': STATUS_OK}
    print(result)
    return result


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
space_raw = {
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
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}
# space = {
#    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
#    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
# }

df_train_sum = pd.read_csv("./features_train_scaled_sum.csv")
df_test_sum = pd.read_csv("./features_test_scaled_sum.csv")
df_y = pd.read_csv("./y_train.csv")

# encode class values as integers so they work as targets for the prediction algorithm
encoder = LabelEncoder()
y = encoder.fit_transform(df_y["surface"])
y_count = len(list(encoder.classes_))
cols_to_drop = ["series_id"]
df_train_sum.drop(cols_to_drop, axis=1, inplace=True)
df_test_sum.drop(cols_to_drop, axis=1, inplace=True)
X = df_train_sum

best = fmin(fn=objective_sklearn,
            space=space,
            algo=tpe.suggest,
            max_evals=10)