#randomforest_autompgregression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

auto_mpg = pd.read_csv(r"D:\4081ml\auto-mpg.csv")
auto1=auto_mpg.drop('car name',axis=1)

print(auto1.isnull().sum())

auto1 = auto1.replace('?', np.nan)

print('\n print how many null values are there in each column')
print(auto1.isnull().sum())

auto1['horsepower']=auto1['horsepower'].astype(float)
print('\n datatypes after conversion of horsepower to float')
print(auto1.dtypes)
print(auto1['horsepower'].describe())

print('\n print mean of each column')
print(auto1.mean())

print('\n after fill null value with mean of the column')
auto2=auto1.fillna(auto1.mean())

X = auto2.drop(['mpg'], axis=1)
y = auto2['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=50, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("mean squared error",mse)
print("R^2 score",r2)
