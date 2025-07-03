#Finding Outliers
import pandas as pd
data=pd.read_csv("Datasets/forestfires.csv")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['month']=le.fit_transform(data['month'])
data['day']=le.fit_transform(data['day'])

X=data.drop(columns=['area'])

from sklearn.preprocessing import  StandardScaler

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

from sklearn.neighbors import LocalOutlierFactor
lof=LocalOutlierFactor(n_neighbors=20,contamination=0.05)
outlier_labels=lof.fit_predict(X_scaled)

data['Outliers']=outlier_labels

print("Number of outlier detected:",sum(outlier_labels==-1))
print(data[data['Outliers']==-1].head())

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.countplot(x='Outliers',data=data)
plt.title("LOF Outlier Detection on Forest fire dataset")
plt.xlabel("Outlier(-1) vs Inlier(1)")
plt.show()