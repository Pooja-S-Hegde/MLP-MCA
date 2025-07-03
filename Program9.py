#Program for Random forest classifier

import pandas as pd
data=pd.read_csv("Datasets/iris_naivebayes.csv")

X= data.drop('target',axis=1)
y=data["target"]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_encoded=le.fit_transform(y)
print("Feature preprocessed")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=(train_test_split(X,y_encoded,test_size=0.2,train_size=0.8,random_state=58))
print("Data split Successful")

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=15,random_state=58)
rf_model.fit(x_train,y_train)

sample_index=10
sample=x_test.iloc[sample_index].values.reshape(1,-1)

tree_pred=[tree.predict(sample)[0] for tree in rf_model.estimators_]

from collections import Counter
vote_count=Counter(tree_pred)

label_votes={le.inverse_transform([int(k)])[0]: v for k,v in vote_count.items()}

print("Class Votes:")
for label,count in label_votes.items():
    print(f"{label}: {count} votes")

majority_encoded,_=vote_count.most_common(1)[0]
majority_label=le.inverse_transform([int(majority_encoded)])[0]

true_label=le.inverse_transform([int(y_test[sample_index])])[0]
print(f"Final Predictions (Majority Vote):{majority_label}")
print(f"Actual Label: {true_label}")