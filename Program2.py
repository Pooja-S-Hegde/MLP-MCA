import pandas as pd
data=pd.read_csv("C:/Users/pr he/OneDrive/Desktop/coding files/MLP-MCA/iris.csv")
print(data.head())
print(data.tail())
print(data.isnull().sum())
X=data.drop("target",axis=1)
y=data["target"]
from sklearn.preprocessing import StandardScaler
Scalar=StandardScaler()
x_scaled=Scalar.fit_transform(X)
print("Features preprocessed")
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,train_size=0.8,random_state=58)
print("Data spit successful")
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
print("Model trained successful")
train_score=model.score(X_train,y_train)
print("Model Accuracy is:",train_score)

import  numpy as np
new_sample=np.array([[5.2,4.8,2.8,8.8]])
new_scaled=Scalar.transform(new_sample)
prediction=model.predict(new_scaled)
print('Target=',prediction[0])
