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
from hyperopt_utils import *
from fit_cv import fit_cv


class LGBMOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    # max number of trials hyperopt runs
    n_trials = 200
    # verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quiet?
    verbosity = 0
    # if true, print summary accuracy/loss after each round
    print_summary = False
    n_classes = 2
    classifier = lgbm.LGBMClassifier
    use_calibration = False

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

        self.fit_params = self.create_fit_params(params)

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        # https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
        space = {
            # this is just piling on most of the possible parameter values for LGBM
            # some of them apparently don't make sense together, but works for now.. :)
            'class_weight': hp.choice('class_weight', [None, 'balanced']),
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt',
                                         #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                         },
                                        # NOTE: DART IS COMMENTED DUE TO SLOW SPEED. HAVE TO MAKE IT AN OPTION..
                                        #                                    {'boosting_type': 'dart',
                                        #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                        #                                     },
                                        {'boosting_type': 'goss'}]),
            'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),  # alias "subsample"
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
            'verbose': -1,
            # the LGBM parameters docs list various aliases, and the LGBM implementation seems to complain about
            # the following not being used due to other params, so trying to silence the complaints by setting to None
            'subsample': None,  # overridden by bagging_fraction
            'reg_alpha': None,  # overridden by lambda_l1
            'reg_lambda': None,  # overridden by lambda_l2
            'min_sum_hessian_in_leaf': None,  # overrides min_child_weight
            'min_child_samples': None,  # overridden by min_data_in_leaf
            'colsample_bytree': None,  # overridden by feature_fraction
            #        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),  # also aliases to min_sum_hessian
            #        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            #        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            #        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        }
        if self.n_classes > 2:
            space['objective'] = "multiclass"
        else:
            space['objective'] = "binary"
            # space["num_class"] = 1
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = None

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)