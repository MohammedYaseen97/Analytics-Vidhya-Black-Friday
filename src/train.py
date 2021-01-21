import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

import config

def run(fold):
    df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train_folds.csv'))
    
    #Product_Category_2 and Product_Category_3 have missing values .. 
    
    df["Product_Category_2"] = df["Product_Category_2"].fillna(np.ceil(df["Product_Category_2"].mean()))
    df["Product_Category_2"] = df["Product_Category_2"].fillna(np.ceil(df["Product_Category_2"].mean()))
    
    cat_cols = ["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3"]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    train = df[df.kfold != fold].reset_index(drop=True)
    val = df[df.kfold == fold].reset_index(drop=True)
    
    train_X = pd.DataFrame({col:train[col].values for col in train.columns if col not in ["kfold", "User_ID", "Product_ID", "Purchase"]})
    train_y = train["Purchase"].values
    
    val_X = pd.DataFrame({col:val[col].values for col in val.columns if col not in ["kfold", "User_ID", "Product_ID", "Purchase"]})
    val_y = val["Purchase"].values
    
    model = config.MODEL
    
    model.fit(train_X, train_y)
    
    preds = model.predict(val_X)
    
    rmse = sqrt(mean_squared_error(val_y, preds))
    
    print("fold = {}, rmse = {}".format(fold, rmse))

if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)