from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data=load_iris()

x=data.data
y=data.target 
xt,tx,yt,ty=train_test_split(x,y,test_size=0.3,random_state=42)
naive_bayes=GaussianNB()

naive_bayes.fit(xt,yt)

yp=naive_bayes.predict(tx)

acc=accuracy_score(ty,yp)
print("accuracy:",acc)
