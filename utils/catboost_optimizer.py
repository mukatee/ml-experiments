__author__ = 'teemu kanstren'

import numpy as np
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import catboost
from sklearn.preprocessing import LabelEncoder
import hyperopt
from test_predictor import stratified_test_prediction_avg_vote
from opt_utils import *
from fit_cv import fit_cv
import pickle

class CatboostOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of rows to use for X and y. to reduce time and compare options faster
    max_n = None
    # max number of trials hyperopt runs
    n_trials = 200
    # verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quite?
    verbosity = 0
    # if true, print summary accuracy/loss after each round
    print_summary = False

    all_accuracies = []
    all_losses = []
    all_params = []

    def objective_sklearn(self, params):
        int_types = ["depth"]
        params = convert_int_params(int_types, params)
        params["iterations"] = 1000
        params["early_stopping_rounds"] = 10
        if params['bootstrap_type'].lower() != "bayesian":
            # catboost gives error if bootstrap option defined with bootstrap disabled
            del params['bagging_temperature']

        #    n_classes = params["num_class"]
        n_classes = params.pop("num_class")

        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, catboost.CatBoostClassifier,
                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)
        self.all_params.append(params)
        self.all_accuracies.append(score)
        self.all_losses.append(logloss)
        if self.verbosity == 0:
            if self.print_summary:
                print("Score {:.3f}".format(score))
        else:
            print("Score {:.3f} params {}".format(score, params))
        # using logloss here for the loss but uncommenting line below calculates it from average accuracy
        #    loss = 1 - score
        loss = logloss
        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
        return result

    def optimize_catboost(self, n_classes, max_n_search):
        # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        # https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
        space = {
            # 'shrinkage': hp.loguniform('shrinkage', -7, 0),
            'depth': hp.quniform('depth', 2, 10, 1),
            'rsm': hp.uniform('rsm', 0.5, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'border_count': hp.qloguniform('border_count', np.log(32), np.log(255), 1),
            # 'ctr_border_count': hp.qloguniform('ctr_border_count', np.log(32), np.log(255), 1),
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0, 5, 1),
            'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
            'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'No']),  # Poisson also possible for GPU
            'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(3)),
            'use_best_model': True
            # 'gradient_iterations': hp.quniform('gradient_iterations', 1, 100, 1),
        }

        self.max_n = max_n_search

        if n_classes > 2:
            space['objective'] = "multiclass"
            space["num_class"] = n_classes
            space["eval_metric"] = "multi_logloss"
        else:
            space['objective'] = "Logloss"
            space["num_class"] = 2
            # space["eval_metric"] = ["Logloss"]
            # space["num_class"] = 1

        trials = Trials()
        best = fmin(fn=self.objective_sklearn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.n_trials,
                    trials=trials)

        # find the trial with lowest loss value. this is what we consider the best one
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

        self.fit_params = {'verbose': self.verbosity,
                           'use_eval_set': True}

        # use 2 classes as this is a binary classification
        # the second param is the number of rows to use for training
        params = self.optimize_catboost(2, 5000)
        print(params)

        clf = catboost.CatBoostClassifier(**params)

        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,
                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)
        search_results.all_accuracies = self.all_accuracies
        search_results.all_losses = self.all_losses
        search_results.all_params = self.all_params
        search_results.best_params = params
        return search_results

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

