#iris_logisticregression

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

iris=load_iris()
x=iris.data
y=iris.target

df=pd.DataFrame(x,columns=iris.feature_names)
df['Spices']=y
print(df.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,
test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=200)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print("Predicted Flower", iris.target_names[prediction][0])
