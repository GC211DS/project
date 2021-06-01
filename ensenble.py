
import pandas as pd
import numpy as np

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder ,StandardScaler)
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, )




df = pd.read_csv("housing.csv")

encoder = LabelEncoder()
ocean_prox = encoder.fit_transform(df['ocean_proximity'])


df.drop('ocean_proximity', axis=1, inplace=True)
df['ocean_proximity'] = ocean_prox

df.fillna(0, inplace=True)

y = df['median_house_value']
df.drop('median_house_value', axis=1, inplace=True)

columns = df.columns

scaler = StandardScaler()
scaled = scaler.fit_transform(df)

df = pd.DataFrame(scaled, columns=columns)

X = df





trainX, testX, trainY, testY = train_test_split(X, y, train_size=4/5, shuffle=True)


# criterion = {mse, friedman_mse, mae, poisson}
# dtreeEntropy1 = DecisionTreeRegressor(max_depth=1, criterion='mse')
# dtreeEntropy2 = DecisionTreeRegressor(max_depth=2, criterion='mse')
dtree = [
    DecisionTreeRegressor(max_depth=3, criterion='mse'),
    DecisionTreeRegressor(max_depth=4, criterion='mse'),
    DecisionTreeRegressor(max_depth=5, criterion='mse'),
    DecisionTreeRegressor(max_depth=6, criterion='mse'),
    DecisionTreeRegressor(max_depth=7, criterion='mse'),
    DecisionTreeRegressor(max_depth=8, criterion='mse'),
    DecisionTreeRegressor(max_depth=9, criterion='mse'),
    DecisionTreeRegressor(max_depth=10, criterion='mse'),
    DecisionTreeRegressor(max_depth=11, criterion='mse'),
    DecisionTreeRegressor(max_depth=12, criterion='mse'),
    DecisionTreeRegressor(max_depth=13, criterion='mse'),
    DecisionTreeRegressor(max_depth=14, criterion='mse'),
    DecisionTreeRegressor(max_depth=15, criterion='mse'),
    DecisionTreeRegressor(max_depth=16, criterion='mse'),
    DecisionTreeRegressor(max_depth=17, criterion='mse'),
    DecisionTreeRegressor(max_depth=18, criterion='mse'),
    DecisionTreeRegressor(max_depth=19, criterion='mse'),
    DecisionTreeRegressor(max_depth=20, criterion='mse'),
    DecisionTreeRegressor(max_depth=25, criterion='mse'),
    DecisionTreeRegressor(max_depth=30, criterion='mse'),
    DecisionTreeRegressor(max_depth=3, criterion='mae'),
    DecisionTreeRegressor(max_depth=4, criterion='mae'),
    DecisionTreeRegressor(max_depth=5, criterion='mae'),
    DecisionTreeRegressor(max_depth=6, criterion='mae'),
    DecisionTreeRegressor(max_depth=7, criterion='mae'),
    DecisionTreeRegressor(max_depth=8, criterion='mae'),
    DecisionTreeRegressor(max_depth=9, criterion='mae'),
    DecisionTreeRegressor(max_depth=10, criterion='mae'),
    DecisionTreeRegressor(max_depth=11, criterion='mae'),
    DecisionTreeRegressor(max_depth=12, criterion='mae'),
    DecisionTreeRegressor(max_depth=13, criterion='mae'),
    DecisionTreeRegressor(max_depth=14, criterion='mae'),
    DecisionTreeRegressor(max_depth=15, criterion='mae'),
    DecisionTreeRegressor(max_depth=16, criterion='mae'),
    DecisionTreeRegressor(max_depth=17, criterion='mae'),
    DecisionTreeRegressor(max_depth=18, criterion='mae'),
    DecisionTreeRegressor(max_depth=19, criterion='mae'),
    DecisionTreeRegressor(max_depth=20, criterion='mae'),
    DecisionTreeRegressor(max_depth=25, criterion='mae'),
    DecisionTreeRegressor(max_depth=30, criterion='mae'),
]

model = BaggingRegressor()


params = {
    'n_estimators': [1000],
    'max_samples': [1], # All samples
    'max_features' : [1], # All features
    'base_estimator' : [DecisionTreeRegressor(max_depth=20, criterion='mse')],
    'bootstrap': [True],
}
grid = GridSearchCV(model, params, cv=5)

grid.fit(X, y)
print("Best params k: ", grid.best_params_)
print("---")
print("(Hypertuned model) Best params k score: ", grid.best_score_)
print("---")
print("grid.best_estimator_: ", grid.best_estimator_)
print("")

bestModel = grid.best_estimator_
best_bootstrap = grid.best_params_['bootstrap']
best_max_features = grid.best_params_['max_features']
best_max_samples = grid.best_params_['max_samples']
best_n_estimators = grid.best_params_['n_estimators']
# mse, mae, mape => 있는 것 사용
# mpe, sst => 직접 구현

model.fit(trainX, trainY)



score = model.score(testX, testY)
print(score)