import numpy as np
import pandas as pd
from sklearn.tree import export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle


df_train = pd.read_csv("train.csv")

df_train.columns = df_train.columns.str.lower()

# translate mssubclass column to actual names
mssubclass_values = {
    20:	"1-STORY 1946 & NEWER ALL STYLES",
    30:	"1-STORY 1945 & OLDER",
    40: "1-STORY W/FINISHED ATTIC ALL AGES",
    45:	"1-1/2 STORY - UNFINISHED ALL AGES",
    50: "1-1/2 STORY FINISHED ALL AGES",
    60:	"2-STORY 1946 & NEWER",
    70:	"2-STORY 1945 & OLDER",
    75:	"2-1/2 STORY ALL AGES",
    80:	"SPLIT OR MULTI-LEVEL",
    85:	"SPLIT FOYER",
    90:	"DUPLEX - ALL STYLES AND AGES",
    120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
    150: "1-1/2 STORY PUD - ALL AGES",
    160: "2-STORY PUD - 1946 & NEWER",
    180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
    190: "2 FAMILY CONVERSION - ALL STYLES AND AGES",
}

df_train.mssubclass = df_train.mssubclass.map(mssubclass_values)

# translate OverallQual, OverallCond column to actual names
overallqual_values = {
       10:	"Very Excellent",
       9:	"Excellent",
       8:	"Very Good",
       7:	"Good",
       6:	"Above Average",
       5:	"Average",
       4:	"Below Average",
       3:	"Fair",
       2:	"Poor",
       1:	"Very Poor",
}
df_train.overallqual = df_train.overallqual.map(overallqual_values)

overallcond_values = {
      10:	"Very Excellent",
       9:	"Excellent",
       8:	"Very Good",
       7:	"Good",
       6:	"Above Average",
       5:	"Average",
       4:	"Below Average",
       3:	"Fair",
       2:	"Poor",
       1:	"Very Poor",
}
df_train.overallcond = df_train.overallcond.map(overallcond_values)

for c in list(df_train.dtypes[df_train.dtypes == "object"].index):
    df_train[c] = df_train[c].str.replace(" ", "_").str.lower()


# dont need the id
df_train = df_train[list(df_train.columns)[1:]]
df_train

# split the train dataset
from sklearn.model_selection import train_test_split
df_train_full, df_train_test = train_test_split(df_train, test_size=0.2, random_state=11)
df_train_train, df_train_val = train_test_split(df_train_full, test_size=0.25,random_state=11)

y_train = np.log1p(df_train_train.saleprice.values)
y_val = np.log1p(df_train_val.saleprice.values)
y_test = np.log1p(df_train_test.saleprice.values)

df_train_train = df_train_train.reset_index(drop=True)
df_train_val = df_train_val.reset_index(drop=True)
df_train_test = df_train_test.reset_index(drop=True)

# delete target value
del df_train_train["saleprice"]
del df_train_val["saleprice"]
del df_train_test["saleprice"]

# fill nan with 0
df_train_train = df_train_train.fillna(0)
df_train_val = df_train_val.fillna(0)

#encode categorical variable
dict_train = df_train_train.to_dict(orient="records")
dict_val = df_train_val.to_dict(orient="records")

"""### XGBoot model is the winner
Train the model with the full train dataset, and test it
"""

# process full train dataset and test
y_train_full = np.log1p(df_train_full.saleprice.values)
df_train_full = df_train_full.reset_index(drop=True)
del df_train_full["saleprice"]
df_train_full = df_train_full.fillna(0)
df_train_test = df_train_test.fillna(0)

# encode categorical variable
dict_train_full = df_train_full.to_dict(orient="records")
dict_train_test = df_train_test.to_dict(orient="records")

# feature matrix
dv = DictVectorizer(sparse=True)
X_train_full = dv.fit_transform(dict_train_full)
X_test = dv.transform(dict_train_test)

# xgb
dfulltrain = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=dv.get_feature_names_out().tolist())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out().tolist())

xgb_params = {
    "eta": 0.3,
    "max_depth": 6,
    "min_child_weight": 10,
    "objective": "reg:squarederror",
    "nthread": 8,
    "seed": 1,
    "silent": 1
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=100)


### Save the model


output_file = "model.bin"

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)


### Load the model

model_file = "model.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

