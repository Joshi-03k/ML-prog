#kmedoid_iris

import numpy as np
import  matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMedoids
from sklearn.metrics import confusion_matrix,classification_report

iris = datasets.load_iris()
X = iris.data
y = iris.target

k = 3  
kmedoids = KMedoids(n_clusters=k, random_state=0).fit(X)

labels = kmedoids.labels_

print("Confusion Matrix:")
print(confusion_matrix(y, labels))
print("\nClassification Report:")
print(classification_report(y, labels))

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('KMedoids Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
