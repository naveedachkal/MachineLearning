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
print(X)
print(y)

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


