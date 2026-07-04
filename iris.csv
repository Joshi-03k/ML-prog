import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.datasets import load_iris

iris=pd.read_csv("D:\kashyap\ML\30_6\iris.csv")

y=iris["species"]
x=iris[["sepal_length","sepal_width", "petal_length","petal_width"]]

print(y)
print(x)
xt,tx,yt,ty=train_test_split(x,y,random_state=50,test_size=0.2)
dtree1=tree.DecisionTreeClassifier(max_depth=3,random_state=1)
dtree1.fit(xt,yt)

yp=dtree1.predict(tx)

print('Accuracy of Decision Tree-test:',accuracy_score(yp,ty))
print('\n',' Confusion Matrix Test:','\n',confusion_matrix(ty,yp))
print(classification_report(ty,yp))
