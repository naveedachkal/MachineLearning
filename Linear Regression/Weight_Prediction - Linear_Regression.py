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
X[:,0]=LabelforGender.fit_transform(X[:,0])
print(X)
