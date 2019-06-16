__author__ = 'teemu kanstren'

import pandas as pd
import numpy as np
import lgbm_optimizer
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
print(df_train.head())

df_train["train"] = 1
df_test["train"] = 0
df_all = pd.concat([df_train, df_test], sort=False)
print(df_all.head())
print(df_all.tail())

y = df_train["Survived"]
y.head()

def parse_cabin_type(x):
    if pd.isnull(x):
        return None
    #print("X:"+x[0])
    return x[0]

def parse_cabin_number(x):
    if pd.isnull(x):
        return -1
#        return np.nan
    cabs = x.split()
    cab = cabs[0]
    num = cab[1:]
    if len(num) < 2:
        return -1
#        return np.nan
    return num

def parse_cabin_count(x):
    if pd.isnull(x):
        return np.nan
    cabs = x.split()
    return len(cabs)

cabin_types = df_all["Cabin"].apply(lambda x: parse_cabin_type(x))
cabin_types = cabin_types.unique()
#drop the nan value from list of cabin types
cabin_types = np.delete(cabin_types, np.where(cabin_types == None))
print(cabin_types)

df_all["cabin_type"] = df_all["Cabin"].apply(lambda x: parse_cabin_type(x))
df_all["cabin_num"] = df_all["Cabin"].apply(lambda x: parse_cabin_number(x))
df_all["cabin_count"] = df_all["Cabin"].apply(lambda x: parse_cabin_count(x))
df_all["cabin_num"] = df_all["cabin_num"].astype(int)
print(df_all.head())

embarked_dummies = pd.get_dummies(df_all["Embarked"], prefix="embarked_", dummy_na=True)
#TODO: see if imputing embardked makes a difference
df_all = pd.concat([df_all, embarked_dummies], axis=1)
print(df_all.head())

cabin_type_dummies = pd.get_dummies(df_all["cabin_type"], prefix="cabin_type_", dummy_na=True)
df_all = pd.concat([df_all, cabin_type_dummies], axis=1)
df_all.head()

l_enc = LabelEncoder()
df_all["sex_label"] = l_enc.fit_transform(df_all["Sex"])
df_all.head()

df_all["family_size"] = df_all["SibSp"] + df_all["Parch"] + 1
df_all.head()

# Cleaning name and extracting Title
for name_string in df_all['Name']:
    df_all['Title'] = df_all['Name'].str.extract('([A-Za-z]+)\.', expand=True)
df_all.head()

print(df_all["Title"].unique())

# Replacing rare titles
mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other',
           'Col': 'Other', 'Dr': 'Other', 'Rev': 'Other', 'Capt': 'Other',
           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal',
           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}

df_all.replace({'Title': mapping}, inplace=True)
# titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']

titles = df_all["Title"].unique()
print(titles)

title_dummies = pd.get_dummies(df_all["Title"], prefix="title", dummy_na=True)
df_all = pd.concat([df_all, title_dummies], axis=1)
print(df_all.head())

titles = list(titles)
# Replacing missing age by median/title
for title in titles:
    age_to_impute = df_all.groupby('Title')['Age'].median()[titles.index(title)]
    df_all.loc[(df_all['Age'].isnull()) & (df_all['Title'] == title), 'Age'] = age_to_impute

print(df_all.groupby('Pclass').agg({'Fare': lambda x: x.isnull().sum()}))

#df_all[df_all["Fare"] == np.nan]
print(df_all[df_all["Fare"].isnull()])

p3_median_fare = df_all[df_all["Pclass"] == 3]["Fare"].median()
print(p3_median_fare)

df_all["Fare"].fillna(p3_median_fare, inplace=True)

df_all['Last_Name'] = df_all['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5
df_all['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
for grp, grp_df in df_all[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in df_all.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 0

df_train = df_all[df_all["train"] == 1]
df_test = df_all[df_all["train"] == 0]

X_cols = set(df_train.columns)
X_cols -= {'PassengerId', 'Survived', 'Sex', 'Name', 'Ticket', 'Cabin', 'Embarked', 'cabin_type', 'Title', 'train', 'Last_Name'}
X_cols = list(X_cols)
print(X_cols)

print(df_train[X_cols].head())

lgbm_optimizer.n_trials = 5
predictions, oof_predictions, _, misclassified_indices = lgbm_optimizer.classify_binary(X_cols, df_train, df_test, y)

df_losses = pd.DataFrame()
df_losses["loss"] = lgbm_optimizer.all_losses
df_losses["accuracy"] = lgbm_optimizer.all_accuracies
df_losses.plot(figsize=(14,8))


print(df_losses.head(10))

print(df_losses.tail(10))

print(df_losses.sort_values(by="loss", ascending=False).head(10))

print(df_losses.sort_values(by="accuracy", ascending=False).head(10))

print(df_losses.sort_values(by="loss", ascending=True).head(10))



ss = pd.read_csv('gender_submission.csv')
# predicting only true values, so take column 1 (0 is false column)
np_preds = np.array(predictions)[:, 1]
ss["Survived"] = np.where(np_preds > 0.5, 1, 0)
ss.to_csv('lgbm.csv', index=False)
ss.head(10)

print(oof_predictions[misclassified_indices])

oof_series = pd.Series(oof_predictions[misclassified_indices])
oof_series.index = y[misclassified_indices].index
print(oof_series)


miss_scale_raw = y[misclassified_indices] - oof_predictions[misclassified_indices]
miss_scale_abs = abs(miss_scale_raw)
miss_scale = pd.concat([miss_scale_raw, miss_scale_abs, oof_series, y[misclassified_indices]], axis=1)
miss_scale.columns = ["Raw_Diff", "Abs_Diff", "Prediction", "Actual"]
miss_scale.head()

df_miss_scale = pd.DataFrame(miss_scale)
df_miss_scale.head()

df_top_misses = df_miss_scale.sort_values(by="Abs_Diff", ascending=False)
df_top_misses.head()

print(df_top_misses.iloc[0:10])

top10 = df_top_misses.iloc[0:10].index
print(top10)

print(df_train.iloc[top10])

print(df_top_misses.iloc[0:10])

df_bottom_misses = df_miss_scale.sort_values(by="Abs_Diff", ascending=True)
df_bottom_misses.head()

bottom10 = df_bottom_misses.iloc[0:10].index
print(bottom10)

print(df_train.iloc[bottom10])


df_bottom_misses = df_miss_scale.sort_values(by="Abs_Diff", ascending=True)
df_bottom_misses.head()

bottom10 = df_bottom_misses.iloc[0:10].index
print(bottom10)

print(df_train.iloc[bottom10])

#if __name__ == "__main__":




