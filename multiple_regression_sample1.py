#multiple_regression_sample1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = {'X1': [1, 2, 3, 4, 5,6,7,8,9,10],
        'X2': [2, 3, 4, 5, 6,7,8,9,10,11],
        'Y': [3, 5, 7, 9, 11,13,15,17,19,21]}

df = pd.DataFrame(data)

X = df[['X1', 'X2']]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print('Mean absolute error:',metrics.mean_absolute_error(Y_test,Y_pred))
print('Mean squared error:',metrics.mean_squared_error(Y_test,Y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred, squared=False))

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
