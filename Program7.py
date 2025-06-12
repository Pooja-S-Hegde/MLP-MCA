#Program to implement Hierarchical Clustering

import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import obrientransform
from seaborn.matrix import dendrogram

data=pd.read_csv("Datasets/ecommerce_customers.csv")

X=data.drop(columns=['CustomerID'])

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
print("Features Scaled")
print(data.head())
print(pd.DataFrame(X_scaled,columns=X.columns).head())

from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=3)
data['Clusters']=model.fit_predict(X_scaled)
print("\nCluster Counts:")
print(data['Clusters'].value_counts().sort_index())

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_pca=pca.fit_transform(X_scaled)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(x=x_pca[:,0], y=x_pca[:, 1], hue=data['Clusters'], palette='Set1')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Hierarchical Clustering")
plt.legend(title="Clusters")
plt.grid(True)
plt.show()

from scipy.cluster.hierarchy import  dendrogram,linkage
linked=linkage(X_scaled,method='ward')
plt.figure(figsize=(10,6))
dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=False)
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()