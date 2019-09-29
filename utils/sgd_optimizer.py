__author__ = 'teemu kanstren'

from sklearn.linear_model import SGDClassifier
from hyperopt_utils import *

class SGDOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of rows to use for X and y. to reduce time and compare options faster
    max_n = None
    # max number of trials hyperopt runs
    n_trials = 200
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    sgd_verbosity = 0
    verbosity = 0

    n_classes = 2
    classifier = SGDClassifier
    use_calibration = True

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def objective_sklearn(self, params):
        int_types = ["n_iter_no_change", "max_iter"]
        params = convert_int_params(int_types, params)
        if params["learning_rate"] == "optimal":
            del params["alpha"]  # alpha cannot be zero in optimal learning rate as it is used as a divider

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        space = {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
            # hinge = linear SVM, log = logistic regression,
            'loss': hp.choice('loss', ['hinge', 'modified_huber', 'squared_hinge', 'perceptron']),
            # https://github.com/scikit-learn/scikit-learn/issues/7278
            # only "log" gives probabilities so have use that. else have to rewrite to allow passing instance,
            # TODO see link above for CalibratedClassifierCV
            # 'loss': hp.choice('loss', ['log']),
            'penalty': hp.choice('penalty', ['none', 'l1', 'l2', 'elasticnet']),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'l1_ratio': hp.choice('l1_ratio', [1, hp.quniform('l1_ratio_fraction', 0.05, 0.95, 0.05)]),
            'max_iter': 1000,
            'n_iter_no_change': 5,
            'early_stopping': True,
            'n_jobs': -1,
            'tol': 0.001,
            'shuffle': True,
            # 'epsilon': ?,
            'learning_rate': hp.choice('learning_rate', ['optimal', 'adaptive']),
            'eta0': 0.001,
            'validation_fraction': 0.1,
            'verbose': self.sgd_verbosity,
            'class_weight': hp.choice('class_weight', ['balanced', None]),
        }
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)