#kmeans_iris_short

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report

iris = datasets.load_iris()
X = iris.data
y = iris.target

k = 3  
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

labels = kmeans.labels_

print("Confusion Matrix:")
print(confusion_matrix(y, labels))
print("\nClassification Report:")
print(classification_report(y, labels))

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
