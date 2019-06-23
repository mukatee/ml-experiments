__author__ = 'teemu kanstren'

import pandas as pd

#check if given parameter can be interpreted as a numerical value
def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

#convert given set of paramaters to integer values
#this at least cuts the excess float decimals if they are there
def convert_int_params(names, params):
    for int_type in names:
        #sometimes the parameters can be choices between options or numerical values. like "log2" vs "1-10"
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params

#convert float parameters to 3 digit precision strings
#just for simpler diplay and all
def convert_float_params(names, params):
    for float_type in names:
        raw_val = params[float_type]
        if is_number(raw_val):
            params[float_type] = '{:.3f}'.format(raw_val)
    return params

def create_misclassified_dataframe(result, y):
    oof_series = pd.Series(result.oof_predictions[result.misclassified_indices])
    oof_series.index = y[result.misclassified_indices].index
    miss_scale_raw = y[result.misclassified_indices] - result.oof_predictions[result.misclassified_indices]
    miss_scale_abs = abs(miss_scale_raw)
    df_miss_scale = pd.concat([miss_scale_raw, miss_scale_abs, oof_series, y[result.misclassified_indices]], axis=1)
    df_miss_scale.columns = ["Raw_Diff", "Abs_Diff", "Prediction", "Actual"]
    result.df_misses = df_miss_scale

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
