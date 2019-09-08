__author__ = 'teemu kanstren'

import numpy as np
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import catboost
from sklearn.preprocessing import LabelEncoder
import hyperopt
from test_predictor import stratified_test_prediction_avg_vote
from hyperopt_utils import *
from fit_cv import fit_cv
import pickle


class CatboostOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of trials hyperopt runs
    n_trials = 200
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    # verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quite?
    verbosity = 0
    # if true, print summary accuracy/loss after each round
    print_summary = False
    use_gpu = False
    n_classes = 2
    classifier = catboost.CatBoostClassifier
    use_calibration = False

    all_accuracies = []
    all_losses = []
    all_params = []

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
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'verbose': self.verbosity,
                           'use_eval_set': True}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)


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

    cat_opt = CatboostOptimizer()

    params = cat_opt.optimize_catboost()
    print(params)

    clf = catboost.CatBoostClassifier(**params)

    search_results = stratified_test_prediction_avg_vote(clf, df_train_sum, df_test_sum, y)
    predictions = search_results["predictions"]

    ss = pd.read_csv('../input/career-con-2019/sample_submission.csv')
    ss['surface'] = encoder.inverse_transform(predictions.argmax(axis=1))
    ss.to_csv('catboost.csv', index=False)
    ss.head(10)
    with open('catboost_params.pickle', 'wb') as handle:
        pickle.dump(search_results.best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

