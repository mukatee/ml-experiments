__author__ = 'teemu kanstren'

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import catboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import hyperopt
import test_predictor
import opt_utils
from sklearn.metrics import accuracy_score

def fit_cv(X, y, params, fit_params):
    score = 0
    n_folds = 5
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        print('-' * 20, i, '-' * 20)

        #https://github.com/Koziev/MNIST_Boosting/blob/master/catboost_hyperopt_solver.py
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        #https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
        clf = catboost.CatBoostClassifier(**params)

        # verbose = print loss at every "verbose" rounds
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100, **fit_params)
        oof_preds = np.zeros((X.shape[0]))
        predictions = clf.predict(X.iloc[test_index])
        oof_preds[test_index] = predictions.flatten()
        iter_score = accuracy_score(oof_preds[test_index], y[test_index])
        score += iter_score
        print('score: {}'.format(iter_score))
        importances = clf.feature_importances_
        features = X.columns
    return score / n_folds


def objective_sklearn(params):
    int_types = ["depth"]
    params = opt_utils.convert_int_params(int_types, params)
    params["objective"] = "MultiClass"
    params["eval_metric"] = "Accuracy"
    params["iterations"] = 1000
    params["early_stopping_rounds"] = 10
    if params['bootstrap_type'].lower() != "bayesian":
        #catboost gives error if bootstrap option defined with bootstrap disabled
        del params['bagging_temperature']

    score = fit_cv(X, y, params, {})
    print("Score {:.3f} params {}".format(score, params))
    loss = 1 - score
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result

def optimize_catboost():
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
    #https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
    space = {
        #'shrinkage': hp.loguniform('shrinkage', -7, 0),
        'depth': hp.quniform('depth', 2, 10, 1),
        'rsm': hp.uniform('rsm', 0.5, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'border_count': hp.qloguniform('border_count', np.log(32), np.log(255), 1),
        #'ctr_border_count': hp.qloguniform('ctr_border_count', np.log(32), np.log(255), 1),
        'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0, 5, 1),
        'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
        'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'No']), #Poisson also possible for GPU
        'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(3)),
        'use_best_model': True
        #'gradient_iterations': hp.quniform('gradient_iterations', 1, 100, 1),
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

    params = optimize_catboost()
    print(params)

    clf = catboost.CatBoostClassifier(**params)

    search_results = test_predictor.stratified_test_prediction_avg_vote(clf, df_train_sum, df_test_sum, y)
    predictions = search_results["predictions"]

    ss = pd.read_csv('../input/career-con-2019/sample_submission.csv')
    ss['surface'] = encoder.inverse_transform(predictions.argmax(axis=1))
    ss.to_csv('catboost.csv', index=False)
    ss.head(10)

