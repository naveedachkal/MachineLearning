"""
@author: Naveed Achkal

"""

#Loading Data
import pandas as pd
df = pd.read_csv("weight-height.csv");

#Analysing data
df.info()
print(df.head())

#Checking Null values
print(df.isnull())
print(df.isnull().sum())

#Dividing the data
X=df.iloc[:, :-1]
y=df.iloc[:, 2]
Height=df.iloc[:,1]
Weight=df.iloc[:,2]
print(X)
print(y)

#Checking co-relation between Height & Weight
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(df.iloc[:,1],df.iloc[:,2])
plt.show()

#Checking for Outliers
sns.boxplot(df.iloc[:,1])
plt.show()
sns.boxplot(df.iloc[:,2])
plt.show()

#Removing Outliers
Ht_q1=df.iloc[:,1].quantile(0.25)
Ht_q3=df.iloc[:,1].quantile(0.75)
Ht_iqr=Ht_q3-Ht_q1
Ht_ul=Ht_q3+1.5*Ht_iqr
Ht_ll=Ht_q1-1.5*Ht_iqr
X=X[(X.iloc[:,1]>=Ht_ll) & (X.iloc[:,1]<=Ht_ul)]

Wt_q1=df.iloc[:,2].quantile(0.25)
Wt_q3=df.iloc[:,2].quantile(0.75)
Wt_iqr=Wt_q3-Wt_q1
Wt_ul=Wt_q3+1.5*Wt_iqr
Wt_ll=Wt_q1-1.5*Wt_iqr
y=y[(y>=Wt_ll) & (y<=Wt_ul)]

#Checking for Outliers
sns.boxplot(X.iloc[:,1])
plt.show()
sns.boxplot(y)
plt.show()

#Reshaping to get equal elements
y = y[:9992]

#Converting Gender to Number
from sklearn.preprocessing import LabelEncoder
LabelforGender= LabelEncoder ()
X.iloc[:,0]=LabelforGender.fit_transform(X.iloc[:,0])
print(X)

#Splitting the data; test & Train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4,random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#Fitting Model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

#Making Prediction
lin_prod=lin_reg.predict(X_test)

#Model Accuracy
from sklearn import metrics
print("R square= ", metrics.r2_score(y_test,lin_prod))
print("MSE= ", metrics.mean_squared_error(y_test,lin_prod))

#Predict Weight
my_pred_weight=lin_reg.predict([[0,58]])
print("My predicted weight=", my_pred_weight)




