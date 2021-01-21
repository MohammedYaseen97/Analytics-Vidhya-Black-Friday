from sklearn import linear_model
import xgboost as xgb

INPUT_PATH=r'../input'
OUTPUT_PATH=r'../input'
MODEL_PATH=r'../models'

#MODEL = linear_model.LinearRegression()
MODEL = xgb.XGBRegressor(verbosity=1)