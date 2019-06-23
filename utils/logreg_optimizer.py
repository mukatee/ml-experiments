__author__ = 'teemu kanstren'

import numpy as np
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import hyperopt
from sklearn.linear_model import LogisticRegression

from fit_cv import fit_cv
from test_predictor import stratified_test_prediction_avg_vote

class LogRegOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of rows to use for training (from X and y). to reduce time and compare options faster
    max_n = None
    # max number of trials hyperopt runs
    n_trials = 200
    # ?
    verbosity = 0
    #if true, print summary accuracy/loss after each round
    print_summary = False

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
        n_classes = params.pop("num_class")
#        params = convert_int_params(int_types, params)
        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, LogisticRegression,
                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)
        self.all_params.append(params)
        self.all_accuracies.append(score)
        self.all_losses.append(logloss)

        loss = logloss
        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
        return result

    def optimize_logreg(self, n_classes):
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
        params = self.optimize_logreg(2)
        print(params)

        clf = LogisticRegression(**params)

        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,
                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)
        search_results.all_accuracies = self.all_accuracies
        search_results.all_losses = self.all_losses
        search_results.all_params = self.all_params
        return search_results
