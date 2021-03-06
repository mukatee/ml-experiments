__author__ = 'teemu kanstren'

import catboost
from optimizer_utils import *

class CatboostOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of trials hyperopt runs
    n_trials = 200
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    cb_verbosity = 0
    verbosity = 0
    use_gpu = False
    n_classes = 2
    classifier = catboost.CatBoostClassifier
    use_calibration = False
    cat_features = None
    scale_pos_weight = None

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def objective_sklearn(self, params):
        int_types = ["depth", "iterations", "early_stopping_rounds"]
        params = convert_int_params(int_types, params)
        if params['bootstrap_type'].lower() != "bayesian":
            # catboost gives error if bootstrap option defined with bootstrap disabled
            del params['bagging_temperature']

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        space = {
            # 'shrinkage': hp.loguniform('shrinkage', -7, 0),
            'depth': hp.quniform('depth', 2, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'border_count': hp.qloguniform('border_count', np.log(32), np.log(255), 1),
            # 'ctr_border_count': hp.qloguniform('ctr_border_count', np.log(32), np.log(255), 1),
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0, 5, 1),
            'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
            'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(3)),
            'use_best_model': True,
            'early_stopping_rounds': 10,
            'iterations': 1000,
            'feature_border_type': hp.choice('feature_border_type',
                                             ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum']),
            # 'gradient_iterations': hp.quniform('gradient_iterations', 1, 100, 1),
        }
        if self.use_gpu:
            space['task_type'] = "GPU"
            space['bootstrap_type'] = hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'Poisson', 'No'])
        else:
            space['task_type'] = "CPU"
            space['rsm'] = hp.uniform('rsm', 0.5, 1)
            space['bootstrap_type'] = hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS', 'No'])

        if self.n_classes > 2:
            space['objective'] = "multiclass"
            space["eval_metric"] = "multi_logloss"
        else:
            space['objective'] = "Logloss"
        if self.scale_pos_weight is not None:
            space["scale_pos_weight"] = self.scale_pos_weight
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'verbose': self.cb_verbosity,
                           'use_eval_set': True}
        if self.cat_features is not None:
            self.fit_params["cat_features"] = self.cat_features

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)
