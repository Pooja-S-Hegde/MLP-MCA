import pandas as pd
data=pd.read_csv("C:/Users/pr he/OneDrive/Desktop/coding files/MLP-MCA/BostonHousing.csv")
print(data.head())
print(data.tail())
print(data.isnull().sum())
X=data[['rm']] #
y=data['medv'] #median values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
#predict
y_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score  

#evaluate
print("R^2:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
import matplotlib.pyplot as plt
plt.scatter(X_test,y_test,color='blue',label='Actual')
plt.plot(X_test,y_pred,color='red',linewidth=2,label='Predicted')
plt.xlabel("Rooms per House")
plt.ylabel("  House Price")
plt.title("Linear Regression")
plt.legend()
plt.show()
