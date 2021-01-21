import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

import config

train_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train.csv'))

test_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'test.csv'))
test_df["Purchase"] = -1

df = pd.concat([train_df, test_df], axis = 0)
    
#Product_Category_2 and Product_Category_3 have missing values .. 

df["Product_Category_2"] = df["Product_Category_2"].fillna(np.ceil(df["Product_Category_2"].mean()))
df["Product_Category_2"] = df["Product_Category_2"].fillna(np.ceil(df["Product_Category_2"].mean()))

cat_cols = ["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3"]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

train = df[df["Purchase"] != -1].reset_index(drop=True)
test = df[df["Purchase"] == -1].reset_index(drop=True)

train_X = pd.DataFrame({col:train[col].values for col in train.columns if col not in ["kfold", "User_ID", "Product_ID", "Purchase"]})
train_y = train["Purchase"].values

test_X = pd.DataFrame({col:test[col].values for col in test.columns if col not in ["kfold", "User_ID", "Product_ID", "Purchase"]})

model = config.MODEL

model.fit(train_X, train_y)

preds = model.predict(test_X)

test["Purchase"] = preds

test.to_csv(os.path.join(config.INPUT_PATH, "submission_xgb.csv"), columns = ["Purchase", "User_ID", "Product_ID"], index = False)