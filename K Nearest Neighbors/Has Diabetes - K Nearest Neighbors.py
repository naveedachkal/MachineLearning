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
df1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(df1.isnull().sum())

#Histogram to understand distribution
p=df.hist(figsize = (20,20))
plt.show()

#imputing values as per their distribution
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

#Count of 'Outcome' by their value
color_wheel = {1: "#0392cf",
               2: "#7bc043"}
colors = df["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(df.Outcome.value_counts())
p=df.Outcome.value_counts().plot(kind="bar")
plt.show()

#Scatter Matrix/Co-relation matrix
from pandas.plotting import scatter_matrix
p3=scatter_matrix(df,figsize = (25,25))
plt.show()

#Pearson's Correlation Coefficient
import seaborn as sns
p4=sns.pairplot(df1)
plt.show()

#Heatmap(two-dimensional representation of information with the help of colors) to understand better
plt.figure(figsize=(50,50))
p6=sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
plt.show()
plt.figure(figsize=(50,50))
p7=sns.heatmap(df1.corr(), annot=True, cmap='RdYlGn')
plt.show()

#Dat
print(df1.head())

#Data Standardisation - As data may have broad range of values so may affect specially while calculating Euclidean distance
from sklearn.preprocessing import StandardScaler
standardised_data=StandardScaler()
X=pd.DataFrame(standardised_data.fit_transform(df1.iloc[:, :-1]))
print(X.head())
y=df1.Outcome
print(y)

#Train-Test split using Stratify - if variable is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=1/3, random_state=3, stratify=y)

#Fitting the Model
from sklearn.neighbors import KNeighborsClassifier
test_scores=[]
train_scores=[]
for i in range(1,15):
       knn=KNeighborsClassifier(i)
       knn.fit(X_train,y_train)
       train_scores.append(knn.score(X_train,y_train))
       test_scores.append(knn.score(X_test,y_test))

#calculating score using test & train data
max_train_score=max(train_scores)
train_scores_index=[i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k={}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_index))))

max_test_score = max(test_scores)
test_scores_index = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_index))))

#Visualisation
plt.figure(figsize=(15,5))
p=sns.lineplot(range(1,15), train_scores, markers='*', label='Train Score')
p=sns.lineplot(range(1,15), test_scores, markers='O', label='Test Score')
plt.show()

#k=11
knn=KNeighborsClassifier(11)
knn.fit(X_train, y_train)
print('Final Accuracy= %', knn.score(X_test, y_test))

#Evaluation - Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


