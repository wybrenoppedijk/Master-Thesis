import pandas as pd
import math as m
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None
import  numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_pickle('../../output/238-va-co3-3600.pkl', compression='gzip')
df_r = df[['prcp', 'inflow']]
df_r['prcp_prev'] = df_r.prcp.shift(1)
df_r['month'] = df_r.index.month
df_r['hour'] = df_r.index.hour
df_r = df_r.reset_index()
df_r = df_r.rename(columns={'index': 'datetime'})
df_r = df_r.dropna()

df_r = pd.get_dummies(df_r, columns=['month', 'hour'])
train, test = df_r[:int(len(df_r)*0.8)], df_r[int(len(df_r)*0.8):]
X, y = df_r.drop(['inflow', 'datetime'], axis=1).to_numpy(), df_r.inflow.to_numpy()
train_X = train.drop(['inflow','datetime'], axis=1).to_numpy()
train_y = train.inflow.to_numpy()
test_X = test.drop(['inflow', 'datetime'], axis=1).to_numpy()
test_y = test.inflow.to_numpy()
scoring = 'neg_mean_squared_error'

parameters = {
    'num_leaves': [50,100,200,300,500],
    'min_child_samples': [5, 10,15,50],
    'learning_rate':[0.05,0.1,0.2, 0.5],
    'reg_alpha':[0,0.01,0.03],
    'n_estimators':[100,200,300,500,1000],
}
gs = RandomizedSearchCV(estimator=lgb.sklearn.LGBMRegressor(), param_distributions=parameters, cv=3, scoring=scoring,
                        random_state=42, n_jobs=14, n_iter=200, verbose=10 )
gs.fit(train_X, train_y)
print(gs.best_params_, gs.best_score_)


# df = df.fillna(0)
# train = df.iloc[0:int(len(df) * 0.8)]
# test = df.iloc[int(len(df) * 0.8):]
# X_train = train[['water_level', 'outflow_level', 'temp', 'current_tot', 'prcp', 'Volume_RolAvg_Vol_7']]
# y_train = train['inflow_label']
# X_test = test[['water_level', 'outflow_level', 'temp', 'current_tot', 'prcp', 'Volume_RolAvg_Vol_7', 'inflow']]
# y_test = test['inflow_label']
# # y_pred = pred_flex(X_test)
# X_test = test[['water_level', 'outflow_level', 'temp', 'current_tot', 'prcp', 'Volume_RolAvg_Vol_7']]
#
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor()
#
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1024, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=10, random_state=42, n_jobs = 14)
# # Fit the random search model
# rf_random.fit(X_train, y_train)
# print(rf_random.best_params_)