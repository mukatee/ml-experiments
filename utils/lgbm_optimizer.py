__author__ = 'teemu kanstren'

import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import hyperopt
from test_predictor import stratified_test_prediction_avg_vote
from opt_utils import *
from fit_cv import fit_cv

class LGBMOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of rows to use for training (from X and y). to reduce time and compare options faster
    max_n = None
    # max number of trials hyperopt runs
    n_trials = 200
    #verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quite?
    verbosity = 0
    #if true, print summary accuracy/loss after each round
    print_summary = False

    from sklearn.metrics import accuracy_score, log_loss

    all_accuracies = []
    all_losses = []
    all_params = []

    def create_fit_params(self, params):
        using_dart = params['boosting_type'] == "dart"
        if params["objective"] == "binary":
            # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
            fit_params = {"eval_metric": ["binary_logloss", "auc"]}
        else:
            fit_params = {"eval_metric": "multi_logloss"}
        if using_dart:
            n_estimators = 2000
        else:
            n_estimators = 15000
            fit_params["early_stopping_rounds"] = 100
        params["n_estimators"] = n_estimators
        fit_params['use_eval_set'] = True
        fit_params['verbose'] = self.verbosity
        return fit_params

    # this is the objective function the hyperopt aims to minimize
    # i call it objective_sklearn because the lgbm functions called use sklearn API
    def objective_sklearn(self, params):
        int_types = ["num_leaves", "min_child_samples", "subsample_for_bin", "min_data_in_leaf"]
        params = convert_int_params(int_types, params)

        # Extract the boosting type
        params['boosting_type'] = params['boosting_type']['boosting_type']
        #    print("running with params:"+str(params))

        fit_params = self.create_fit_params(params)
        if params['objective'] == "binary":
            n_classes = 2
        else:
            n_classes = params["num_class"]

        score, logloss = fit_cv(self.X, self.y, params, fit_params, n_classes, lgbm.LGBMClassifier,
                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)
        self.all_params.append(params)
        self.all_accuracies.append(score)
        self.all_losses.append(logloss)
        if self.verbosity == 0:
            if self.print_summary:
                print("Score {:.3f}".format(score))
        else:
            print("Score {:.3f} params {}".format(score, params))
    #using logloss here for the loss but uncommenting line below calculates it from average accuracy
    #    loss = 1 - score
        loss = logloss
        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
        return result

    def optimize_lgbm(self, n_classes):
        # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        # https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
        space = {
            #this is just piling on most of the possible parameter values for LGBM
            #some of them apparently don't make sense together, but works for now.. :)
            'class_weight': hp.choice('class_weight', [None, 'balanced']),
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt',
    #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                         },
                                        {'boosting_type': 'dart',
    #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                         },
                                        {'boosting_type': 'goss'}]),
            'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
            'verbose': -1,
            #the LGBM parameters docs list various aliases, and the LGBM implementation seems to complain about
            #the following not being used due to other params, so trying to silence the complaints by setting to None
            'subsample': None, #overridden by bagging_fraction
            'reg_alpha': None, #overridden by lambda_l1
            'reg_lambda': None, #overridden by lambda_l2
            'min_sum_hessian_in_leaf': None, #overrides min_child_weight
            'min_child_samples': None, #overridden by min_data_in_leaf
            'colsample_bytree': None, #overridden by feature_fraction
    #        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5), #also aliases to min_sum_hessian
    #        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    #        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    #        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        }
        if n_classes > 2:
            space['objective'] = "multiclass"
            space["num_class"] = n_classes
        else:
            space['objective'] = "binary"
            #space["num_class"] = 1

        trials = Trials()
        best = fmin(fn=self.objective_sklearn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.n_trials,
                    trials=trials,
                   verbose= 1)

        # find the trial with lowest loss value. this is what we consider the best one
        idx = np.argmin(trials.losses())
        print(idx)

        print(trials.trials[idx])

        # these should be the training parameters to use to achieve the best score in best trial
        params = trials.trials[idx]["result"]["params"]
        max_n = None

        print(params)
        return params

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param):
        self.y = y_param

        self.X = df_train[X_cols]
        self.X_test = df_test[X_cols]

        # use 2 classes as this is a binary classification
        params = self.optimize_lgbm(2)
        print(params)

        clf = lgbm.LGBMClassifier(**params)

        fit_params = self.create_fit_params(params)

        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,
                                                             n_folds=self.n_folds, n_classes=2, fit_params=fit_params)
        search_results.all_accuracies = self.all_accuracies
        search_results.all_losses = self.all_losses
        search_results.all_params = self.all_params
        return search_results
