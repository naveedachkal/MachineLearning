import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

#Loading the data
df=pd.read_csv("diabetes.csv")
print(df.head())
print(df.describe())
df.info()
print(df.describe().T)

#No True Zero value; Hence we will replace 0 with NaN
df1=df.copy(deep=True)
df1.iloc[:, :-1]=df1.iloc[:, :-1].replace(0,np.NaN)
print(df1.isnull().sum())

#Histogram to understand distribution
p=df.hist(figsize = (20,20))
plt.show()

#imputing values as per their distribution
df1['Pregnancies'].fillna(df1['Pregnancies'].mean(), inplace=True)
df1['Glucose'].fillna(df1['Glucose'].mean(), inplace=True)
df1['BloodPressure'].fillna(df1['BloodPressure'].mean(), inplace=True)
df1['SkinThickness'].fillna(df1['SkinThickness'].median(), inplace=True)
df1['Insulin'].fillna(df1['Insulin'].median(), inplace=True)
df1['BMI'].fillna(df1['BMI'].median(), inplace=True)

#After NaN removal
p1=df1.hist(figsize = (20,20))
plt.show()

#Null count Analysis
import missingno as msno
p2=msno.bar(df)
plt.show()

#Scatter Matrix/Co-relation matrix
from pandas.plotting import scatter_matrix
p3=scatter_matrix(df,figsize = (20,20))
plt.show()
import seaborn as sns
p4=sns.pairplot(df1)
plt.show()


