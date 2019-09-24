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


class OptimizerResult:
    avg_accuracy = None,
    misclassified_indices = None,
    misclassified_expected = None,
    misclassified_actual = None,
    oof_predictions = None,
    predictions = None,
    df_misses = None,
    all_accuracies = None,
    all_losses = None,
    all_params = None,
    best_params = None,
    # the final classifier created from optimized hyperparameters, to allow feature weight access etc
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


def stratified_test_prediction_avg_vote(parent, clf, X_train, X_test, y, n_folds, n_classes,
                                        fit_params, train_indices = None, use_calibration = False):
    folds = StratifiedKFold(n_splits = n_folds, shuffle = True)#, random_state = 69)
    #for results. N columns, one per target label. each contains probability of that value
    #sub_preds is for submission predictions, for the "real" test
    sub_preds = np.zeros((X_test.shape[0], n_classes))
    #oof is for out-of-fold predictions. CV predictions over the training data
    oof_preds = np.zeros((X_train.shape[0]))
    use_eval_set = fit_params.pop("use_eval_set")
    acc_score = 0
    acc_score_total = 0
    misclassified_indices = []
    misclassified_expected = []
    misclassified_actual = []
    X_train_full = X_train
    # cut the data if asked to train only on a subset of the training data
    if train_indices is not None:
        X_train = X_train.iloc[train_indices]
        y = y.iloc[train_indices]
        X_train = X_train.reset_index(drop = True)
        y = y.reset_index(drop = True)

    #some classifiers like SGD require calibratedclassifier to give probabilities
    if use_calibration:
        clf = CalibratedClassifierCV(clf, cv = 5, method = 'sigmoid')

    for i, (train_index, test_index) in enumerate(folds.split(X_train, y)):
        print('-' * 20, i, '-' * 20)

        X_val, y_val = X_train.iloc[test_index], y[test_index]
        if use_eval_set:
            clf.fit(X_train.iloc[train_index], y[train_index], eval_set = ([(X_val, y_val)]), **fit_params)
        else:
            #some classifiers, e.g., random forest, do not know parameter "eval_set" or "verbose"
            clf.fit(X_train.iloc[train_index], y[train_index], **fit_params)
        #using predict_probe instead of predict to get probabilities that can be converted or not as needed
        oof_preds[test_index] = clf.predict_proba(X_train.iloc[test_index])[:, 1].flatten()
        #if we were asked to train only on part of the data, predict on the left-out part as well
        #to produce a full result set in the end
        if train_indices is not None:
            extra_test_index = X_train_full.drop(train_indices).index
            oof_preds[extra_test_index] = clf.predict_proba(X_train_full.iloc[extra_test_index])[:, 1].flatten()

        #n folds, so going to add only n:th fraction here
        sub_preds += clf.predict_proba(X_test) / folds.n_splits
        preds_this_round = oof_preds[test_index] >= 0.5
        acc_score = accuracy_score(y[test_index], preds_this_round)
        acc_score_total += acc_score
        print('accuracy score ', acc_score)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            features = X_train.columns

            feat_importances = pd.Series(importances, index = features)
            #feat_importances.nlargest(50).sort_values().to_frame().to_csv(f"top_features_{i}.csv")
            feat_importances.sort_values(ascending=True).to_frame().to_csv(f"top_features_{i}.csv")
            feat_importances.nlargest(30).sort_values().plot(kind = 'barh', color = '#86bf91', figsize = (10, 8))
            plt.show()
        else:
            print("classifier has no feature importances: skipping feature plot")

        missed = y[test_index] != preds_this_round
        misclassified_indices.extend(test_index[missed])
        #this is the set we expected to predict
        m1 = y[test_index][missed]
        misclassified_expected.append(m1)
        #this is the set we actually predicted
        m2 = oof_preds[test_index][missed].astype("int")
        misclassified_actual.append(m2)
        if hasattr(parent, 'cleanup'):
            #if the parent optimizer implements iteration cleanup, call it
            parent.cleanup(clf)

    #print(f"acc_score: {acc_score}")
    sub_sub = sub_preds[:5]
    print(f"predictions on the submisson set first 5 rows: {sub_sub}")
    avg_accuracy = acc_score_total / folds.n_splits
    print('Avg Accuracy', avg_accuracy)
    #not we store the actual results in a single object and return that
    result = OptimizerResult()
    result.avg_accuracy = avg_accuracy
    result.misclassified_indices = misclassified_indices
    result.misclassified_expected = misclassified_expected
    result.misclassified_actual = misclassified_actual
    result.oof_predictions = oof_preds
    result.predictions = sub_preds
    create_misclassified_dataframe(result, y)
    return result

def create_misclassified_dataframe(result, y):
    oof_series = pd.Series(result.oof_predictions[result.misclassified_indices])
    oof_series.index = y[result.misclassified_indices].index
    miss_scale_raw = y[result.misclassified_indices] - result.oof_predictions[result.misclassified_indices]
    miss_scale_abs = abs(miss_scale_raw)
    df_miss_scale = pd.concat([miss_scale_raw, miss_scale_abs, oof_series, y[result.misclassified_indices]], axis = 1)
    df_miss_scale.columns = ["Raw_Diff", "Abs_Diff", "Prediction", "Actual"]
    result.df_misses = df_miss_scale


# run n_folds of cross validation on the data
# averages fold results
def fit_cv(parent, X, y, params, fit_params, n_classes, classifier, use_calibration,
           n_folds, print_summary, verbosity, train_indices):
    X_full = X
    y_full = y
    # cut the data if max_n is set
    if train_indices is not None:
        X = X.iloc[train_indices]
        y = y.iloc[train_indices]

    X = X.reset_index(drop = True)
    y = y.reset_index(drop = True)
    print(X.shape)
    fit_params = fit_params.copy()
    use_eval = fit_params.pop("use_eval_set")
    acc_score = 0
    folds = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 69)

    if print_summary:
        print(f"Running {n_folds} folds...")
    oof_preds = np.zeros((X_full.shape[0], n_classes))
    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        if verbosity > 0:
            print('-' * 20, f"RUNNING FOLD: {i}/{n_folds}", '-' * 20)

        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        clf = classifier(**params)
        if use_calibration:
            clf = CalibratedClassifierCV(clf, cv = 5, method = 'sigmoid')
        # verbose = print loss at every "verbose" rounds.
        # if 100 it prints progress 100,200,300,... iterations
        if use_eval:
            clf.fit(X_train, y_train, eval_set = (X_test, y_test), **fit_params)
        else:
            clf.fit(X_train, y_train, **fit_params)
        oof_preds[test_index] = clf.predict_proba(X.iloc[test_index])
        if train_indices is not None:
            extra_indices = X_full.drop(train_indices).index
            oof_preds[extra_indices] = clf.predict_proba(X_full.iloc[extra_indices])

        acc_score += accuracy_score(y[test_index], oof_preds[test_index][:, 1] >= 0.5)
        if hasattr(parent, 'cleanup'):
            parent.cleanup(clf)
    # accuracy is calculated each fold so divide by n_folds.
    # not n_folds -1 because it is not sum by row but overall sum of accuracy of all test indices
    total_acc_score = acc_score / n_folds
    logloss = log_loss(y_full, oof_preds)
    if print_summary:
        print(f"total acc: {total_acc_score}, logloss={logloss}")
    return total_acc_score, logloss


# run n_folds of cross validation on the data
# averages fold results
def run_iteration(parent, X, y, params, fit_params, n_classes, classifier, use_calibration,
                  n_folds, print_summary, verbosity):
    print(X.shape)
    fit_params = fit_params.copy()
    use_eval = fit_params.pop("use_eval_set")
    acc_score = 0

    if print_summary:
        print(f"Running search over data size {X.shape[0]}...")
    test_set_size = 1 / n_folds

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set_size)

    clf = classifier(**params)
    if use_calibration:
        clf = CalibratedClassifierCV(clf, cv = 5, method = 'sigmoid')
    # verbose = print loss at every "verbose" rounds.
    # if 100 it prints progress 100,200,300,... iterations
    if use_eval:
        clf.fit(X_train, y_train, eval_set = (X_test, y_test), **fit_params)
    else:
        clf.fit(X_train, y_train, **fit_params)
    preds = clf.predict_proba(X_test)

    acc_score += accuracy_score(y_test, preds[:, 1] >= 0.5)
    if hasattr(parent, 'cleanup'):
        parent.cleanup(clf)

    total_acc_score = acc_score
    logloss = log_loss(y_test, preds)
    if print_summary:
        print(f"total acc: {total_acc_score}, logloss={logloss}")
    return total_acc_score, logloss


def hyperopt_run_search(parent):
    space = parent.create_hyperspace()
    trials = Trials()
    best = fmin(fn = parent.objective_sklearn,
                space = space,
                algo = tpe.suggest,
                max_evals = parent.n_trials,
                trials = trials)

    # find the trial with lowest loss value. this is what we consider the best one
    idx = np.argmin(trials.losses())
    print(idx)

    print(trials.trials[idx])

    params = trials.trials[idx]["result"]["params"]
    print(params)
    return params, idx


def hyperopt_search_classify(parent, X_cols, df_train, df_test, y_param, train_pct, stratify_train):
    parent.y = y_param
    parent.X = df_train[X_cols]
    parent.X_test = df_test[X_cols]
    # convert train_pct into percentage to drop, i.e. the test set from split

    # resetting index since sliced dataframes can cause issues in stratified split indexing later
    # this might be something to consider in special cases
    parent.X = parent.X.reset_index(drop = True)
    parent.X_test = parent.X_test.reset_index(drop = True)
    parent.y = parent.y.reset_index(drop = True)

    if train_pct is not None:
        test_pct = 1 - train_pct
        train_indices, test_indices = train_test_split(parent.X.index, test_size = test_pct, stratify = stratify_train)
        parent.train_indices = train_indices

    params, idx = hyperopt_run_search(parent)
    print(params)

    clf = parent.classifier(**params)

    search_results = stratified_test_prediction_avg_vote(parent, clf, parent.X, parent.X_test, parent.y, n_folds = parent.n_folds,
                                                         n_classes = parent.n_classes, fit_params = parent.fit_params,
                                                         train_indices = parent.train_indices, use_calibration = parent.use_calibration)
    search_results.all_accuracies = parent.all_accuracies
    search_results.all_losses = parent.all_losses
    search_results.all_params = parent.all_params
    search_results.best_params = params
    search_results.clf = clf
    return search_results


def hyperopt_objective_run(parent, params, use_cv = False):
    if use_cv:
        score, logloss = fit_cv(parent, parent.X, parent.y, params, parent.fit_params, parent.n_classes, parent.classifier,
                                parent.use_calibration, parent.n_folds, parent.print_summary, verbosity = parent.verbosity,
                                train_indices = parent.train_indices)
    else:
        score, logloss = run_iteration(parent, parent.X, parent.y, params, parent.fit_params, parent.n_classes, parent.classifier,
                                       parent.use_calibration, parent.n_folds, parent.print_summary, verbosity = parent.verbosity)
    parent.all_params.append(params)
    parent.all_accuracies.append(score)
    parent.all_losses.append(logloss)
    if parent.verbosity == 0:
        if parent.print_summary:
            print("Score {:.3f}".format(score))
    else:
        print("Score {:.3f} params {}".format(score, params))
    # using logloss here for the loss but uncommenting line below calculates it from average accuracy
    #    loss = 1 - score
    loss = logloss
    result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}
    return result

