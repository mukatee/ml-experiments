__author__ = 'teemu kanstren'

import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import hyperopt

from test_predictor import stratified_test_prediction_avg_vote
from fit_cv import fit_cv
from opt_utils import *

class RFOptimizer:
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
        int_types = ["n_estimators", "min_samples_leaf", "min_samples_split", "max_features"]
        n_classes = params.pop("num_class")
        params = convert_int_params(int_types, params)
        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, RandomForestClassifier,
                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)
        self.all_params.append(params)
        self.all_accuracies.append(score)
        self.all_losses.append(logloss)

        loss = logloss
        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
        return result

    def optimize_rf(self, n_classes):
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
            'n_jobs': -1,
            'num_class': n_classes,
            'verbose': self.verbosity
        }
        # save and reload trials for hyperopt state: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/hyperopt_optimize.py

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
        params = self.optimize_rf(2)
        print(params)

        clf = RandomForestClassifier(**params)

        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,
                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)
        search_results.all_accuracies = self.all_accuracies
        search_results.all_losses = self.all_losses
        search_results.all_params = self.all_params
        return search_results
