#Program to support vector machine

import pandas as pd
data=pd.read_csv("Datasets/give_me_credit.csv")
data=data.dropna()

X=data.drop("SeriousDlqin2yrs",axis=1)
y=data["SeriousDlqin2yrs"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=58)
print("Data spit successful")

from sklearn.svm import SVC
model=SVC(kernel="rbf",C=1.0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification report:",classification_report(y_test,y_pred))

correct=X_test[y_pred==y_test]
wrong=X_test[y_pred!=y_test]

print("Correct Predictions:")
print(correct.head())

print("Wrong Predictions:")
print(wrong.head())