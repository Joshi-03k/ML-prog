#kmeans_breastcancer_short

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_data)

df['Cluster'] = kmeans.labels_

plt.scatter(df['mean radius'], df['mean texture'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('K-means Clustering of Breast Cancer Data')
plt.show()
