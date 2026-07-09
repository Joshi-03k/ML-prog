
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

spam_data = pd.read_csv('spambase.csv')

X = spam_data[['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
          'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
          'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
          'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
          'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu',
          'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']]
y = spam_data['spam']


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
