__author__ = 'teemu kanstren'

import numpy as np
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import xgboost as xgb
import hyperopt
from opt_utils import *
from fit_cv import fit_cv
from test_predictor import stratified_test_prediction_avg_vote

class XGBOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of rows to use for training (from X and y). to reduce time and compare options faster
    max_n = None
    # max number of trials hyperopt runs
    n_trials = 200
    #verbosity 0 in RF is quite, 1 = print epoch, 2 = print within epoch
    #https://stackoverflow.com/questions/31952991/what-does-the-verbosity-parameter-of-a-random-forest-mean-sklearn
    verbosity = 0
    #if true, print summary accuracy/loss after each round
    print_summary = False

    all_accuracies = []
    all_losses = []
    all_params = []

    def objective_sklearn(self, params):
        int_params = ['max_depth']
        params = convert_int_params(int_params, params)
        float_params = ['gamma', 'colsample_bytree']
        params = convert_float_params(float_params, params)
        n_classes = params.pop("num_class")

        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, xgb.XGBClassifier,
                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)
        self.all_params.append(params)
        self.all_accuracies.append(score)
        self.all_losses.append(logloss)

        loss = logloss
        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
        return result

    def optimize_xgb(self, n_classes):
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
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)]),
            'num_class': n_classes,
            'verbose': self.verbosity
            #'n_estimators': 1000   #n_estimators = n_trees -> get error this only valid for gbtree
            #https://github.com/dmlc/xgboost/issues/3789
        }

        trials = Trials()
        best = fmin(fn=self.objective_sklearn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.n_trials,
                   trials=trials)

        idx = np.argmin(trials.losses())
        print(idx)

        print(trials.trials[idx])

        params = trials.trials[idx]["result"]["params"]
        print(params)
        return params

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param):
        self.y = y_param

        self.X = df_train[X_cols]
        self.X_test = df_test[X_cols]

        self.fit_params = {'use_eval_set': False}

        # use 2 classes as this is a binary classification
        params = self.optimize_xgb(2)
        print(params)

        clf = xgb.XGBClassifier(**params)

        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,
                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)
        search_results.all_accuracies = self.all_accuracies
        search_results.all_losses = self.all_losses
        search_results.all_params = self.all_params
        search_results.best_params = params
        return search_results