#agglomerative_titanic

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv(r"D:\4081\titanic.csv")
print("Dataset shape:", df.shape)

num_df = df.select_dtypes(include=[np.number])

num_df = num_df.fillna(num_df.mean())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clust.fit_predict(scaled_data)

df['Cluster'] = labels

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=labels, cmap='viridis', s=50)
plt.title('Agglomerative Clustering on Titanic Dataset (PCA projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

linked = linkage(scaled_data, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title('Dendrogram (Hierarchical Clustering)')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

df.to_csv("titanic_clusters.csv", index=False)
print("Clustered dataset saved as 'titanic_clusters.csv'")
