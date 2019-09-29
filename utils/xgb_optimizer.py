__author__ = 'teemu kanstren'

import numpy as np
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import xgboost as xgb
import hyperopt
from hyperopt_utils import *
from fit_cv import fit_cv
from test_predictor import stratified_test_prediction_avg_vote

class XGBOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of trials hyperopt runs
    n_trials = 200
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    xgb_verbosity = 0
    verbosity = 0
    # if true, print summary accuracy/loss after each round
    print_summary = False
    n_classes = 2
    classifier = xgb.XGBClassifier
    use_calibration = False
    scale_pos_weight = None

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def cleanup(self, clf):
        # print("cleaning up..")
        # TODO: run in different process.. : https://stackoverflow.com/questions/56298728/how-do-i-free-all-memory-on-gpu-in-xgboost
        clf._Booster.__del__()
        import gc
        gc.collect()

    #        print(dir(clf))
    #        clf.del()

    def objective_sklearn(self, params):
        int_params = ['max_depth']
        params = convert_int_params(int_params, params)
        float_params = ['gamma', 'colsample_bytree']
        params = convert_float_params(float_params, params)

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        space = {
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            # removed gblinear since it does not support early stopping and it was getting tricky
            'booster': hp.choice('booster', ['gbtree']),  # , 'dart']),
            # 'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            # nthread defaults to maximum so not setting it
            'subsample': hp.uniform('subsample', 0.75, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
            # 'gamma': hp.uniform('gamma', 0.0, 0.5),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)]),
            'verbose': self.xgb_verbosity,
            'n_jobs': 4,
            # tree_method: 'gpu_hist' causes xgboost to use GPU. comment in/out if needed
            # https://xgboost.readthedocs.io/en/latest/gpu/
            'tree_method': 'gpu_hist',
            # 'n_estimators': 1000   #n_estimators = n_trees -> get error this only valid for gbtree
            # https://github.com/dmlc/xgboost/issues/3789
        }
        if self.scale_pos_weight is not None:
            space["scale_pos_weight"] = self.scale_pos_weight
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)