__author__ = 'teemu kanstren'

import numpy as np
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import hyperopt
from sklearn.linear_model import LogisticRegression
from hyperopt_utils import *

from fit_cv import fit_cv
from test_predictor import stratified_test_prediction_avg_vote

class LogRegOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    # max number of trials hyperopt runs
    n_trials = 200
    n_classes = 2
    verbosity = 0
    #if true, print summary accuracy/loss after each round
    print_summary = False
    classifier = LogisticRegression
    use_calibration = False

    all_accuracies = []
    all_losses = []
    all_params = []

    def objective_sklearn(self, params):
        #print(params)
        params.update(params["solver_params"]) #pop nested dict to top level
        del params["solver_params"] #delete the original nested dict after pop (could pop() above too..)
        if params["penalty"] == "none":
            del params["C"]
            del params["l1_ratio"]
        elif params["penalty"] != "elasticnet":
            del params["l1_ratio"]
        if params["solver"] == "liblinear":
            params["n_jobs"] = 1
        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        space = {
            'solver_params': hp.choice('solver_params', [
                {'solver': 'newton-cg',
                 'penalty': hp.choice('penalty-ncg', ["l2", 'none'])}, #also multiclass loss supported
                 {'solver': 'lbfgs',
                 'penalty': hp.choice('penalty-lbfgs', ["l2", 'none'])},
                 {'solver': 'liblinear',
                 'penalty': hp.choice('penalty-liblin', ["l1", "l2"])},
                 {'solver': 'sag',
                 'penalty': hp.choice('penalty-sag', ["l2", 'none'])},
                 {'solver': 'saga',
                 'penalty': hp.choice('penalty-saga', ["elasticnet", "l1", "l2", 'none'])},
            ]),
            'C': hp.uniform('C', 1e-5,10),
            'tol': hp.uniform('tol', 1e-5, 10),
            'fit_intercept': hp.choice("fit_intercept", [True, False]),
            'class_weight': hp.choice("class_weight", ["balanced", None]),
            #multi-class jos ei bianry
            'l1_ratio': hp.uniform('l1_ratio', 0.00001, 0.99999), #vain jos elasticnet penalty
            'n_jobs': -1,
            'num_class': self.n_classes,
            'verbose': self.verbosity
        }
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)
