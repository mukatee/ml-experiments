__author__ = 'teemu kanstren'

from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def stratified_test_prediction_avg_vote(clf, X_train, X_test, y, use_eval_set):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
    #9 columns, one per target label. each contains probability of that value
    sub_preds = np.zeros((X_test.shape[0], 9))
    oof_preds = np.zeros((X_train.shape[0]))
    score = 0
    misclassified_indices = []
    misclassified_tuples_all = []
    misclassified_expected = []
    misclassified_actual = []
    for i, (train_index, test_index) in enumerate(folds.split(X_train, y)):
        print('-' * 20, i, '-' * 20)

        X_val, y_val = X_train.iloc[test_index], y[test_index]
        if use_eval_set:
            clf.fit(X_train.iloc[train_index], y[train_index], eval_set=([(X_val, y_val)]), verbose=0)
        else:
            #random forest does not know parameter "eval_set" or "verbose"
            clf.fit(X_train.iloc[train_index], y[train_index])
        oof_preds[test_index] = clf.predict(X_train.iloc[test_index]).flatten()
        sub_preds += clf.predict_proba(X_test) / folds.n_splits
        score += clf.score(X_train.iloc[test_index], y[test_index])
        print('score ', clf.score(X_train.iloc[test_index], y[test_index]))
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            features = X_train.columns

            feat_importances = pd.Series(importances, index=features)
            feat_importances.nlargest(30).sort_values().plot(kind='barh', color='#86bf91', figsize=(10, 8))
            plt.show()
        else:
            print("classifier has no feature importances: skipping feature plot")

        missed = y[test_index] != oof_preds[test_index]
        misclassified_indices.append(test_index[missed])
        m1 = y[test_index][missed]
        misclassified_expected.append(m1)
        m2 = oof_preds[test_index][missed].astype("int")
        misclassified_actual.append(m2)

    avg_accuracy = score / folds.n_splits
    print('Avg Accuracy', avg_accuracy)
    result = {
        "avg_accuracy": avg_accuracy,
        "misclassified_indices": misclassified_indices,
        "misclassified_samples_expected": misclassified_expected,
        "misclassified_samples_actual": misclassified_actual,
        "predictions": sub_preds
    }
    return result


def full_train_predict(clf, X_train, X_test, y):
    pass

def avg_vote(predictions):
    pass

def avg_multi_vote(predictions):
    pass

def majority_vote(predictions):
    pass

