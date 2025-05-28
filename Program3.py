import pandas as pd

from Program2 import prediction

data=pd.read_csv("C:/Users/pr he/OneDrive/Desktop/coding files/MLP-MCA/PlayTennis.csv")
print(data.head())
print(data.isnull().sum())
print(data.info())
print(data.describe())
for col in data.columns[:-1]:
    data[col]=data[col].astype('category')
    mapping=dict(enumerate(data[col].cat.categories))
    print(f"{col}:{mapping}")
    data[col]=data[col].cat.codes
    print("Categorical to numerical conversion successful")
target='Play Tennis'
data[target]=data[target].map({'Yes':1,'No':0})
print("Target conversion successful")

from sklearn.model_selection import train_test_split
X=data.drop(target,axis=1)
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=58)
print("Data spit successful")

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy',random_state=58)
model.fit(X_train,y_train)
print("Model trained successful")
print("\nTraining Accuracy is:",model.score(X_train,y_train))
print("\nTesting Accuracy is:",model.score(X_test,y_test))
sample=pd.DataFrame([[2,1,0,2]],columns=X.columns)
prediction=model.predict(sample)
print("Prediction=",prediction)
