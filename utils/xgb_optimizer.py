__author__ = 'teemu kanstren'

#TODO: consider usefulnes of nested CV: https://stackoverflow.com/questions/52138897/fitting-in-nested-cross-validation-with-cross-val-score-with-pipeline-and-gridse/52147410#52147410

"""
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/:
objective [default=reg:linear]

    This defines the loss function to be minimized. Mostly used values are:
        binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
        multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
            you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
        multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.

eval_metric [ default according to objective ]

    The metric to be used for validation data.
    The default values are rmse for regression and error for classification.
    Typical values are:
        rmse – root mean square error
        mae – mean absolute error
        logloss – negative log-likelihood
        error – Binary classification error rate (0.5 threshold)
        merror – Multiclass classification error rate
        mlogloss – Multiclass logloss
        auc: Area under the curve


"""


import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import xgboost as xgb
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

        clf = xgb.XGBClassifier(**params)
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        #is eval_set is a tuple and not a list of tuples, get weird key error 0
        #https://stackoverflow.com/questions/52338532/keyerror-0-with-xgboost-scikit-learn-and-pandas
        clf.fit(X_train, y_train, eval_metric="mlogloss", eval_set=[(X_test, y_test)], verbose=0, **fit_params)
        oof_preds = np.zeros((X.shape[0]))
        oof_preds[test_index] = clf.predict(X.iloc[test_index])
        score += clf.score(X.iloc[test_index], y[test_index])
        print('score ', clf.score(X.iloc[test_index], y[test_index]))
        if params["booster"] != "dart":
            #xgb dart booster does not define feature importances
            importances = clf.feature_importances_
        features = X.columns
    return score / n_folds


def objective_sklearn(params):
    print("params")
    int_params = ['max_depth']
    params = opt_utils.convert_int_params(int_params, params)
    float_params = ['gamma', 'colsample_bytree']
    params = opt_utils.convert_float_params(float_params, params)
    print("params:"+str(params))
    print("x-shape:"+str(X.shape))

    fit_params = {}
    fit_params["early_stopping_rounds"] = 10
    score = fit_cv(X, y, params, fit_params)
    print("Score {:.3f} params {}".format(score, params))
    loss = 1 - score
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result

def optimize_xgb():
    #https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
    space = {
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        #removed gblinear since it does not support early stopping and it was getting tricky
        'booster': hp.choice('booster', ['gbtree', 'dart']),
        #'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        #nthread defaults to maximum so not setting it
        'subsample': hp.uniform('subsample', 0.75, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
        #'gamma': hp.uniform('gamma', 0.0, 0.5),
        'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
        'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
        'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
        'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
        #'n_estimators': 1000   #n_estimators = n_trees -> get error this only valid for gbtree
        #https://github.com/dmlc/xgboost/issues/3789
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

    params = optimize_xgb()
    print(params)

    clf = xgb.XGBClassifier(**params)

    search_results = test_predictor.stratified_test_prediction_avg_vote(clf, df_train_sum, df_test_sum, y)
    predictions = search_results["predictions"]

    ss = pd.read_csv('./sample_submission.csv')
    ss['surface'] = encoder.inverse_transform(predictions.argmax(axis=1))
    ss.to_csv('lgbm.csv', index=False)
    ss.head(10)

