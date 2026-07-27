#Multinomial Logistic Regression on Glass 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("D:\4081ml\glass.csv")

print("First Five Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names")
print(df.columns)

if 'Id' in df.columns:
    df = df.drop('Id', axis=1)


X = df.drop('Type', axis=1)
y = df['Type']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))


sample = [[1.518,13.6,3.5,1.3,72.7,0.5,8.7,0.0,0.1]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("\nPredicted Glass Type:", prediction[0])


probability = model.predict_proba(sample)

print("\nPrediction Probabilities")
print(probability)
