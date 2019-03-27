__author__ = 'teemu kanstren'

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def create_classifier(params):
    clf = RandomForestClassifier(**params)
    return clf


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

for i, (train_index, test_index) in enumerate(folds.split(X, y)):
    print('-' * 20, i, '-' * 20)

    clf = create_classifier({"n_estimators":200, "n_jobs":-1})
    clf.fit(X.iloc[train_index], y[train_index])
    sub_preds = np.zeros((X_test.shape[0], 9))
    oof_preds = np.zeros((X.shape[0]))
    oof_preds[test_index] = clf.predict(X.iloc[test_index])
    sub_preds += clf.predict_proba(X) / folds.n_splits
    score += clf.score(X.iloc[test_index], y[test_index])
    print('score ', clf.score(X.iloc[test_index], y[test_index]))
    importances = clf.feature_importances_
    features = X.columns

