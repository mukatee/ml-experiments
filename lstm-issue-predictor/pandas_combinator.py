__author__ = 'teemu kanstren'

import glob, os
import pandas as pd

folder_of_download = os.getcwd()
listing = glob.glob(folder_of_download + '/bugs2019/bugs-*.csv')
listing = sorted(listing)
#read all non-empty data-files as separate dataframes
dataframes = [pd.read_csv(file, parse_dates=["Created"]) for file in listing if os.stat(file).st_size > 10]
big_df = pd.concat(dataframes)
#nans = big_df.isnull().sum()
#non_nans = big_df.notnull().sum()
#for index, value in nans.iteritems():
#	print(index, "=", value)
#big_df.isnull()
#df['e'] = df.sum(axis=1)

ids = big_df["Issue key"]
ids.head()
#find and remove duplicates of downloaded issues. sometimes after download restart or error it can happen
dups = big_df[ids.isin(ids[ids.duplicated()])].sort_values(by="Issue key")
print("dups shape: "+str(dups.shape))
big_df = big_df.drop_duplicates(subset="Issue key", keep="first")

#save the full dataset as a single big dataframe (csv file)
big_df.to_csv("all_data.csv")

all_cols = big_df.columns

#this dataset has many very sparse columns, where comment is a different field, etc.
#this just sums all these comment1, comment2, comment3, ... as a count of such items
def sum_columns(old_col_name, new_col_id):
	old_cols = [col for col in all_cols if old_col_name in col]
	olds_df = big_df[old_cols]
	olds_non_null = olds_df.notnull().sum(axis=1)
	big_df.drop(old_cols, axis=1, inplace=True)
	big_df[new_col_id] = olds_non_null

big_df["comp1"] = big_df["Component/s"]
big_df["comp2"] = big_df["Component/s.1"]
sum_columns("Outward issue", "outward_count")
sum_columns("Custom field", "custom_count")
sum_columns("Comment", "comment_count")
sum_columns("Component", "component_count")
sum_columns("Labels", "labels_count")
sum_columns("Affects Version", "affects_count")
sum_columns("Attachment", "attachment_count")
sum_columns("Fix Version", "fix_version_count")
sum_columns("Log Work", "log_work_count")

#owi_cols = [col for col in all_cols if "Outward issue" in col]
#owi_df = big_df[owi_cols]
#owni_non_null = owi_df.notnull().sum(axis=1)
#big_df.drop(owi_cols, axis=1, inplace=True)
#big_df["outward_count"] = owni_non_null

#print indices of any remaining rows with null/nan values
nans = big_df.isnull().sum()
non_nans = big_df.notnull().sum()
for index, value in nans.iteritems():
	print(index, "=", value)

#the final dataframe (csv) i used. reduced to fewer columns
big_df.to_csv("reduced.csv")

#what is Custom field (Rank) ? strange values

#big_df["Reporter"].value_counts()

