import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
import hyperopt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV


# a class to hold the results of the optimizer and final classifier run
class OptimizerResult:
    avg_accuracy = None,
    # indices of misclassified items/rows in the training set for the final run
    misclassified_indices = None,
    # the misclassified rows in the training data, so this is the values that should have been predicted
    misclassified_expected = None,
    # the actual misclassified rows, so the values the were predicted but should not have been
    misclassified_actual = None,
    # out of fold - predictions. CV predictions over the training data.
    oof_predictions = None,
    # test set predictions
    predictions = None,
    # dataframe with some statistics over which items/rows were misclassified and how much
    df_misses = None,
    # list of accuracies of each hyperopt iteration
    all_accuracies = None,
    # logloss of each hyperopt iteration
    all_losses = None,
    # parameters of each hyperopt iteration. the ones selected by hyperopt for the iteration
    all_params = None,
    # dataframe to hold parameters and metrics over all iterations
    iteration_df = None,
    # how much time did it take for each optimization iteration to run
    iteration_times = [],
    # the "best" parameters found by hyperopt over all iterations (lowest loss)
    best_params = None,
    # the final classifier created from optimized hyperparameters, used to run the final test set prediction, to allow feature weight access etc
    # this is created with the "best" parameters, trained on the training set, and used to predict on test set
    clf = None,


# check if given parameter can be interpreted as a numerical value
def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


# convert given set of paramaters to integer values
# cuts float decimals if any
def convert_int_params(names, params):
    for int_type in names:
        # sometimes the parameters can be choices between options or numerical values. like "log2" vs "10"
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params


# convert float parameters to 3 digit precision strings
# just for simpler diplay and all
def convert_float_params(names, params):
    for float_type in names:
        raw_val = params[float_type]
        if is_number(raw_val):
            params[float_type] = '{:.3f}'.format(raw_val)
    return params


def predict_in_batches(clf, df_data, n_classes, verbosity, start):
    size = df_data.shape[0]
    vprint(verbosity, start, f"size in predict_batches: {size}, classes={n_classes}")
    preds = np.zeros((size, n_classes))
    idx1 = 0
    batch_size = 50000
    while idx1 < size:
        idx2 = idx1 + batch_size
        # TODO: test if overflow in one
        preds[idx1:idx2] = clf.predict_proba(df_data[idx1:idx2])
        idx1 = idx2
    return preds

#just print "msg" using a given verbosity setting
def vprint(verbosity, start_time, msg):
    if verbosity > 1:
        end_time = time.time()
        diff = end_time - start_time
        print(f"delta from timestamp: {diff}s")
    if verbosity > 0:
        print(msg)


# run n-fold training and prediction on the train and test data
# params:
# parent:  the optimizer for specific classifier. see this repo/other scripts for examples
# clf:     the actual classifier instance to use

# def stratified_test_prediction_avg_vote(parent, clf, X_train, X_test, y, n_folds, n_classes,
#                                        fit_params, train_indices=None, use_calibration=False):
def stratified_test_prediction_avg_vote(parent, clf):
    start = time.time()
    n_folds = parent.n_folds
    X_test = parent.X_test
    X_train = parent.X
    y = parent.y
    n_classes = parent.n_classes
    fit_params = parent.fit_params
    train_indices = parent.train_indices
    use_calibration = parent.use_calibration
    verbosity = parent.verbosity

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True)  # , random_state = 69)
    # for results. N columns, one per target label. each contains probability of that value
    # sub_preds is for submission predictions, for the "real" test
    sub_preds = np.zeros((X_test.shape[0], n_classes))
    # oof is for out-of-fold predictions. CV predictions over the training data
    oof_preds = np.zeros((X_train.shape[0]))
    use_eval_set = fit_params.pop("use_eval_set")
    acc_score_total = 0
    misclassified_indices = []
    misclassified_expected = []
    misclassified_actual = []
    X_train_full = X_train
    # cut the data if asked to train only on a subset of the training data
    if train_indices is not None:
        X_train = X_train.iloc[train_indices]
        y = y.iloc[train_indices]
        X_train = X_train.reset_index(drop=True)
        y = y.reset_index(drop=True)

    # some classifiers like SGD require calibratedclassifier to give probabilities
    if use_calibration:
        clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')

    for i, (train_index, test_index) in enumerate(folds.split(X_train, y)):
        print('-' * 20, i, '-' * 20)

        X_val, y_val = X_train.iloc[test_index], y[test_index]
        X_sub_train = X_train.iloc[train_index]
        X_sub_y = y[train_index]
        vprint(verbosity, start, f"fitting on train: {X_sub_train.shape} rows, val: {X_sub_y.shape}")
        preds = do_fit(clf, use_calibration, use_eval_set, X_sub_train, y[train_index], X_val, y_val, fit_params, n_classes, verbosity, start_time=start)

        # using predict_proba instead of predict to get probabilities that can be converted or not as needed
        oof_preds[test_index] = preds[:, 1].flatten()
        # if we were asked to train only on part of the data, predict on the left-out part as well
        # to produce a full result set in the end
        if train_indices is not None:
            extra_test_index = X_train_full.drop(train_indices).index
            vprint(verbosity, start, f"doing extra predictions for left-over training data: {len(extra_test_index)} rows")
            extra_train_data = X_train_full.iloc[extra_test_index]
            extra_preds = predict_in_batches(clf, extra_train_data, n_classes, verbosity, start)[:, 1].flatten()
            #extra_preds = clf.predict_proba(extra_train_data)[:, 1].flatten()
            oof_preds[extra_test_index] += extra_preds / n_folds

        # n folds, so going to add only n:th fraction here
        vprint(verbosity, start, f"predicting on the validation set: {X_test.shape} rows")
        sub_preds += predict_in_batches(clf, X_test, n_classes, verbosity, start) / n_folds
        #sub_preds += clf.predict_proba(X_test) / folds.n_splits
        preds_this_round = oof_preds[test_index] >= 0.5
        acc_score = accuracy_score(y[test_index], preds_this_round)
        acc_score_total += acc_score
        print('accuracy score ', acc_score)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            features = X_train.columns

            feat_importances = pd.DataFrame()
            feat_importances["weight"] = importances
            feat_importances.index = features
            feat_importances.sort_values(by="weight", ascending=False).to_csv(f"top_features_{i}.csv")
            feat_importances.nlargest(30, ["weight"]).sort_values(by="weight").plot(kind='barh', title=f"top features {i}", color='#86bf91', figsize=(10, 8))
            # kaggle shows output image files (like this png) under "output visualizations", others (such as pdf) under "output"
            plt.savefig(f'feature-weights-{i}.png')
            plt.savefig(f'feature-weights-{i}.pdf')
            plt.show()
        else:
            print("classifier has no feature importances: skipping feature plot")

        missed = y[test_index] != preds_this_round
        misclassified_indices.extend(test_index[missed])
        # this is the set we expected to predict
        m1 = y[test_index][missed]
        misclassified_expected.append(m1)
        # this is the set we actually predicted
        m2 = oof_preds[test_index][missed].astype("int")
        misclassified_actual.append(m2)
        if hasattr(parent, 'cleanup'):
            # if the parent optimizer implements iteration cleanup, call it
            vprint(verbosity, start, "parent cleanup")
            parent.cleanup(clf)

    # print(f"acc_score: {acc_score}")
    sub_sub = sub_preds[:5]
    vprint(verbosity, start, f"predictions on the submisson set first 5 rows: {sub_sub}")
    avg_accuracy = acc_score_total / folds.n_splits
    print(f'Avg Accuracy {avg_accuracy}')
    # not we store the actual results in a single object and return that
    result = OptimizerResult()
    result.avg_accuracy = avg_accuracy
    result.misclassified_indices = misclassified_indices
    result.misclassified_expected = misclassified_expected
    result.misclassified_actual = misclassified_actual
    result.oof_predictions = oof_preds
    result.predictions = sub_preds
    result.clf = clf
    create_misclassified_dataframe(result, y)
    return result


# create a dataframe with statistics on which rows were misclassified. add it to the given results object.
# params:
# result: OptimizerResults instance, already initialized with raw misclassification data
# y: the "real" target values that should have been predicted
def create_misclassified_dataframe(result, y):
    oof_series = pd.Series(result.oof_predictions[result.misclassified_indices])
    oof_series.index = y[result.misclassified_indices].index
    miss_scale_raw = y[result.misclassified_indices] - result.oof_predictions[result.misclassified_indices]
    miss_scale_abs = abs(miss_scale_raw)
    df_miss_scale = pd.concat([miss_scale_raw, miss_scale_abs, oof_series, y[result.misclassified_indices]], axis=1)
    df_miss_scale.columns = ["Raw_Diff", "Abs_Diff", "Prediction", "Actual"]
    result.df_misses = df_miss_scale


# run n_folds of cross validation on the data
# averages fold results
# params:
# parent: parent optimizer instance
# params: constructor parameters for classifier used
# start_time: for tracking execution time for logs

def fit_cv(parent, params, start_time=time.time()):
    n_folds = parent.n_folds
    X = parent.X
    y = parent.y
    n_classes = parent.n_classes
    fit_params = parent.fit_params
    train_indices = parent.train_indices
    use_calibration = parent.use_calibration
    classifier = parent.classifier
    verbosity = parent.verbosity

    X_full = X
    y_full = y
    # cut the data if max_n is set
    if train_indices is not None:
        X = X.iloc[train_indices]
        y = y.iloc[train_indices]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    vprint(verbosity, start_time, X.shape)
    fit_params = fit_params.copy()
    use_eval = fit_params.pop("use_eval_set")
    acc_score = 0
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    print(f"Running {n_folds} folds...")
    oof_preds = np.zeros((X_full.shape[0], n_classes))
    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        print('-' * 20, f"RUNNING FOLD: {i}/{n_folds}", '-' * 20)

        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        clf = classifier(**params)

        vprint(verbosity, start_time, f"starting to fit in fit_cv")
        preds = do_fit(clf, use_calibration, use_eval, X_train, y_train, X_test, y_test, fit_params, n_classes, verbosity, start_time)
        oof_preds[test_index] = preds

        if train_indices is not None:
            extra_indices = X_full.drop(train_indices).index
            vprint(verbosity, start_time, f"starting to predict extra train in fit_cv: {len(extra_indices)} rows")
            oof_preds[extra_indices] += predict_in_batches(clf, X_full.iloc[extra_indices], n_classes, verbosity, start_time)
            #oof_preds[extra_indices] += clf.predict_proba(X_full.iloc[extra_indices]) / n_folds

        acc_score += accuracy_score(y[test_index], oof_preds[test_index][:, 1] >= 0.5)
        if hasattr(parent, 'cleanup'):
            parent.cleanup(clf)
    vprint(verbosity, start_time, f"fit_cv done")
    # accuracy is calculated each fold so divide by n_folds.
    # not n_folds -1 because it is not sum by row but overall sum of accuracy of all test indices
    total_acc_score = acc_score / n_folds
    logloss = log_loss(y_full, oof_preds)
    print(f"total acc: {total_acc_score}, logloss={logloss}")
    return total_acc_score, logloss


# fit a given classifier using given configuration / parameters
def do_fit(clf, use_calibration, use_eval, X_train, y_train, X_test, y_test, fit_params, n_classes, verbosity, start_time):
    if use_calibration:
        clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    vprint(verbosity, start_time, f"fitting in do_fit, eval={use_eval}, data size={X_train.shape}")
    if use_eval:
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), **fit_params)
    else:
        clf.fit(X_train, y_train, **fit_params)
    vprint(verbosity, start_time, f"predicting in do_fit {X_test.shape} data shape")
    preds = predict_in_batches(clf, X_test, n_classes, verbosity, start_time)
    #preds = clf.predict_proba(X_test)
    vprint(verbosity, start_time, "done predicting in do_fit")
    return preds


# run n_folds of cross validation on the data
# averages fold results
# parent = parent optimizer instance for a specific classifier,
# params = parameters to give to actual classifier constructor when creating object
def run_iteration(parent, params, start_time):
    n_folds = parent.n_folds
    X = parent.X
    y = parent.y
    fit_params = parent.fit_params
    use_calibration = parent.use_calibration
    classifier = parent.classifier
    verbosity = parent.verbosity

    # cut the data if max_n is set
    if parent.train_indices is not None:
        X = X.iloc[parent.train_indices]
        y = y.iloc[parent.train_indices]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    vprint(verbosity, start_time, X.shape)
    fit_params = fit_params.copy()
    use_eval = fit_params.pop("use_eval_set")
    acc_score = 0

    vprint(verbosity, start_time, f"Running search over data size {X.shape[0]}...")
    test_set_size = 1 / n_folds

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size)

    clf = classifier(**params)
    preds = do_fit(clf, use_calibration, use_eval, X_train, y_train, X_test, y_test, fit_params, parent.n_classes, verbosity, start_time)

    acc_score += accuracy_score(y_test, preds[:, 1] >= 0.5)
    if hasattr(parent, 'cleanup'):
        parent.cleanup(clf)

    total_acc_score = acc_score
    logloss = log_loss(y_test, preds)
    print(f"total acc: {total_acc_score}, logloss={logloss}")
    return total_acc_score, logloss


# run the overall search
def hyperopt_run_search(parent):
    space = parent.create_hyperspace()
    trials = Trials()
    best = fmin(fn=parent.objective_sklearn,
                space=space,
                algo=tpe.suggest,
                max_evals=parent.n_trials,
                trials=trials)

    # find the trial with lowest loss value. this is what we consider the best one
    idx = np.argmin(trials.losses())
    print("best iteration", idx)

    print(trials.trials[idx])

    params = trials.trials[idx]["result"]["params"]
    print(params)
    return params, idx


# this is the actual method to start the optimization process
# params:
# parent: the optimizer instance for a specific classifier being optimized. check the classes in this repo/other scipts
# X_cols: set of columns/features to pick from the training data for training.
# df_train: the training data. will be split to train/validation sets
# df_test: separate test set to predict on at the end
# y: the target variable data
# train_pct: the percentage of training data to use. if None, use all. if 0-1, use that percentage. If > 1 or < 0, likely crash.
# stratify_train: if defined, stratify the train/test split on this data. typically this would be equal to "y" param (target).
def hyperopt_search_classify(parent, X_cols, df_train, df_test, y, train_pct, stratify_train):
    # store parameter values in parent optimizer object, as that is available to use when hyperopt starts a classifier iteration
    parent.y = y
    parent.X = df_train[X_cols]
    parent.X_test = df_test[X_cols]

    # resetting index since sliced dataframes can cause issues in stratified split indexing later
    # this might be something to consider in special cases
    parent.X = parent.X.reset_index(drop=True)
    parent.X_test = parent.X_test.reset_index(drop=True)
    parent.y = parent.y.reset_index(drop=True)

    # convert train_pct into percentage to drop, i.e. the test set from split
    if train_pct is not None:
        test_pct = 1 - train_pct
        train_indices, test_indices = train_test_split(parent.X.index, test_size=test_pct, stratify=stratify_train)
        parent.train_indices = train_indices

    params, idx = hyperopt_run_search(parent)
    print(params)

    clf = parent.classifier(**params)

    search_results = stratified_test_prediction_avg_vote(parent, clf)
    search_results.all_accuracies = parent.all_accuracies
    search_results.all_losses = parent.all_losses
    search_results.all_params = parent.all_params
    search_results.all_times = parent.all_times
    search_results.best_params = params
    search_results.clf = clf

    df_rows = []
    for idx, val in enumerate(search_results.all_params):
        param_dict = val.copy()
        df_rows.append(param_dict)
    iteration_df = pd.DataFrame(df_rows)
    iteration_df["g_accuracy"] = search_results.all_accuracies
    iteration_df["g_loss"] = search_results.all_losses
    iteration_df["g_duration"] = search_results.all_times
    search_results.iteration_df = iteration_df
    iteration_df.to_csv("iterations.csv")

    return search_results


# called from the optimizer from the objective_sklearn() function to proceed with running the hyperopt iteration
# parent = optimizer parent
# params = constructor parameters for classifier
# use_cv = if true, run this iteration using n-fold cross-validatiom. else just on train-test split. generally much faster.
def hyperopt_objective_run(parent, params, use_cv=False):
    start_time = time.time()
    if use_cv:
        score, logloss = fit_cv(parent, params, start_time)
    else:
        score, logloss = run_iteration(parent, params, start_time)
    end = time.time()
    diff = end - start_time
    parent.all_params.append(params)
    parent.all_accuracies.append(score)
    parent.all_losses.append(logloss)
    parent.all_times.append(diff)

    vprint(parent.verbosity, start_time, "Score {:.3f}".format(score))

    # using logloss here for the loss but uncommenting line below calculates it from average accuracy
    #    loss = 1 - score
    #or replace with anything else you want to "minimize"
    loss = logloss
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result

