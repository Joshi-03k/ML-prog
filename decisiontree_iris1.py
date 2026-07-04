from sklearn import datasets
import sklearn.model_selection as ms
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 


data=datasets.load_breast_cancer()
xt,tx,yt,ty=ms.train_test_split(data.data,data.target,test_size=0.2,random_state=42)
dtc=tree.DecisionTreeClassifier(random_state=42)
dtc.fit(xt,yt)

yp=dtc.predict(tx)

print('Accuracy if Decision Tree',accuracy_score(yp,ty))
print('\n','Confusion Matric - Test:','\n',confusion_matrix(ty,yp))
print(classification_report(ty,yp))
