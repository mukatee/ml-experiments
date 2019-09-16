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
from hyperopt_utils import *

class RFOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    n_classes = 2
    # max number of trials hyperopt runs
    n_trials = 200
    #verbosity 0 in RF is quite, 1 = print epoch, 2 = print within epoch
    #https://stackoverflow.com/questions/31952991/what-does-the-verbosity-parameter-of-a-random-forest-mean-sklearn
    verbosity = 0
    #if true, print summary accuracy/loss after each round
    print_summary = False
    classifier = RandomForestClassifier
    use_calibration = False

    all_accuracies = []
    all_losses = []
    all_params = []

    def objective_sklearn(self, params):
        int_types = ["n_estimators", "min_samples_leaf", "min_samples_split", "max_features"]
        params = convert_int_params(int_types, params)
        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
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
            'verbose': self.verbosity
        }
        # save and reload trials for hyperopt state: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/hyperopt_optimize.py

        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)
