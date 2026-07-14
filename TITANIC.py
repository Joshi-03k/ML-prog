from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd


# Load the Breast Cancer dataset

titanic_data = pd.read_csv('titanic.csv')

titanic_data = titanic_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

titanic_data["Age"].fillna(titanic_data["Age"].median(), inplace=True)

titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Embarked"], drop_first=True)


X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)
svc_model=SVC(kernel='linear',C=1,gamma=10)
dtc = tree.DecisionTreeClassifier(random_state=42)
naive_bayes = GaussianNB()
for k in range(6,17):
    # Instantiate learning model (k = 6 to 16)
    classifier = KNeighborsClassifier(n_neighbors=k)


# fit the model
svc_model.fit(X_train, y_train)
log_reg_model = LogisticRegression()
rf_classifier.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)
dtc.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)
classifier.fit(X_train, y_train)



# Predict the labels for test set
y_predRF = rf_classifier.predict(X_test)
y_predLo = log_reg_model.predict(X_test)
y_predSV = svc_model.predict(X_test)
y_predictDTC=dtc.predict(X_test)
y_prednb=naive_bayes.predict(X_test)
y_predKNN = classifier.predict(X_test)

# Calculate the accuracy
accuracyRF = accuracy_score(y_test, y_predRF)
accuracyLO = accuracy_score(y_test, y_predLo)
accuracySV = accuracy_score(y_test, y_predSV)
accuracyNB = accuracy_score(y_test, y_prednb)
accuracyKNN = accuracy_score(y_test, y_predKNN)
print("Accuracy Of KNN :", accuracyKNN)
print("Accuracy Of naive byres :", accuracyNB)
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_predictDTC, y_test))
print("Accuracy Of randomforest :", accuracyRF)
print("Accuracy of Logistic Regression :", accuracyLO)
print("Accuracy Of SVM :", accuracySV)


# Print classification report
print("\nClassification Report of Random forest:")
print(classification_report(y_test, y_predRF))

print("\nClassification Report Of knn :")
print(classification_report(y_test, y_predKNN))
print("\nClassification Report of Decesion tree:")
print(classification_report(y_test, y_predictDTC))
print("\nClassification Report of naive byres:")
print(classification_report(y_test, y_prednb))
print("\nClassification Report logistic regression:")
print(classification_report(y_test, y_predLo))
print("\nClassification Report SVM:")
print(classification_report(y_test, y_predSV))



