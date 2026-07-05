from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


data=load_breast_cancer()
x=data.data
y=data.target
xt,tx,yt,ty=train_test_split(x,y,test_size=0.2,random_state=42)
naive_bayes=GaussianNB()

naive_bayes.fit(xt,yt)
yp=naive_bayes.predict(tx)

accuracy=accuracy_score(ty,yp)
print("accuracy:",accuracy)