__author__ = 'teemu kanstren'

from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from hyperopt_utils import OptimizerResult, create_misclassified_dataframe
from logreg_optimizer import LogRegOptimizer

from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


def stratified_test_prediction_avg_vote(parent, clf, X_train, X_test, y, n_folds, n_classes, fit_params, train_indices = None, use_calibration = False):
    folds = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 69)
    # N columns, one per target label. each contains probability of that value
    sub_preds = np.zeros((X_test.shape[0], n_classes))
    oof_preds = np.zeros((X_train.shape[0]))
    use_eval_set = fit_params.pop("use_eval_set")
    score = 0
    acc_score = 0
    acc_score_total = 0
    misclassified_indices = []
    misclassified_expected = []
    misclassified_actual = []
    X_train_full = X_train
    y_full = y
    # cut the data if max_n is set
    if train_indices is not None:
        X_train = X_train.iloc[train_indices]
        y = y.iloc[train_indices]
        X_train = X_train.reset_index(drop = True)
        y = y.reset_index(drop = True)
    if use_calibration:
        clf = CalibratedClassifierCV(clf, cv = 5, method = 'sigmoid')
    for i, (train_index, test_index) in enumerate(folds.split(X_train, y)):
        print('-' * 20, i, '-' * 20)

        X_val, y_val = X_train.iloc[test_index], y[test_index]
        if use_eval_set:
            clf.fit(X_train.iloc[train_index], y[train_index], eval_set = ([(X_val, y_val)]), **fit_params)
        else:
            # random forest does not know parameter "eval_set" or "verbose"
            clf.fit(X_train.iloc[train_index], y[train_index], **fit_params)
        # could directly do predict() here instead of predict_proba() but then mismatch comparison would not be possible
        oof_preds[test_index] = clf.predict_proba(X_train.iloc[test_index])[:, 1].flatten()
        if train_indices is not None:
            extra_test_index = X_train_full.drop(train_indices).index
            oof_preds[extra_test_index] = clf.predict_proba(X_train_full.iloc[extra_test_index])[:, 1].flatten()

        # we predict on whole test set, thus split by n_splits, not n_splits - 1
        sub_preds += clf.predict_proba(X_test) / folds.n_splits
        #        sub_preds += clf.predict(X_test) / folds.n_splits
        #        score += clf.score(X_train.iloc[test_index], y[test_index])
        preds_this_round = oof_preds[test_index] >= 0.5
        acc_score = accuracy_score(y[test_index], preds_this_round)
        acc_score_total += acc_score
        print('accuracy score ', acc_score)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            features = X_train.columns

            feat_importances = pd.Series(importances, index = features)
            feat_importances.nlargest(50).sort_values().to_frame().to_csv(f"top_features_{i}.csv")
            feat_importances.nlargest(30).sort_values().plot(kind = 'barh', color = '#86bf91', figsize = (10, 8))
            plt.show()
        else:
            print("classifier has no feature importances: skipping feature plot")

        missed = y[test_index] != preds_this_round
        misclassified_indices.extend(test_index[missed])
        m1 = y[test_index][missed]
        misclassified_expected.append(m1)
        m2 = oof_preds[test_index][missed].astype("int")
        misclassified_actual.append(m2)
        if hasattr(parent, 'cleanup'):
            parent.cleanup(clf)

    print(f"acc_score: {acc_score}")
    sub_sub = sub_preds[:5]
    print(f"sub_preds: {sub_sub}")
    avg_accuracy = acc_score_total / folds.n_splits
    print('Avg Accuracy', avg_accuracy)
    result = OptimizerResult()
    result.avg_accuracy = avg_accuracy
    result.misclassified_indices = misclassified_indices
    result.misclassified_expected = misclassified_expected
    result.misclassified_actual = misclassified_actual
    result.oof_predictions = oof_preds
    result.predictions = sub_preds
    return result


#example of ensemble input:
#capture the probabilities of True (1) classification for each classifier, to use as inputs for ensembling:
#ensemble_input_df = pd.DataFrame()
#ensemble_input_df["lgbm"] = lgbm_results.predictions[:,1]
#ensemble_input_df["xgb"] = xgb_results.predictions[:,1]
#ensemble_input_df["catboost"] = cb_results.predictions[:,1]
#ensemble_input_df["randomforest"] = rf_results.predictions[:,1]
#ensemble_input_df.head()

def ensemble_avg(ensemble_input_df, col_names):
    sum = 0
    for col_name in col_names:
        sum += ensemble_input_df[col_name]
    avg = sum / len(col_names)
    ensemble_input_df["avg"] = avg
    result = np.where(ensemble_input_df["avg"] > 0.5, 1, 0)
    return result

def ensemble_majority(ensemble_input_df, col_names):
    from scipy.stats import mode

    data = [ensemble_input_df[col_name] for col_name in col_names]
    majority = mode(data, axis=0)
    return majority

def ensemble_stack(X_cols, df_train, df_test, target):
    logreg_opt = LogRegOptimizer()
    lr_results = logreg_opt.classify_binary(X_cols, df_train, df_test, target)
    create_misclassified_dataframe(lr_results, target)
    return lr_results

