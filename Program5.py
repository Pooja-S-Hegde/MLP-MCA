#Program to implement K-nearest neighbor Algorithm to classify

import pandas as pd
data=pd.read_csv("Datasets/iris_naivebayes.csv")
print(data.head())
print(data.isnull().sum())
print(data.info())
print(data.describe())

X= data.drop('target',axis=1)
y=data["target"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
print("Feature preprocessed")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=(train_test_split(x_scaled,y,test_size=0.2,train_size=0.8,random_state=58))
print("Data split Successful")

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

train_accuracy=knn.score(x_train,y_train)
test_accuracy=knn.score(x_test,y_test)
print("Training Accuracy is:",train_accuracy)
print("Testing Accuracy is:",test_accuracy)

y_pred=knn.predict(x_test)
correct_pred=(y_pred==y_test)
wrong_pred=(y_pred!=y_test)

print("Correct Predictions:")
print(x_test[correct_pred])

print("Wrong Predictions:")
print(x_test[wrong_pred])