#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:20:17 2018

@author: changyueh
"""

#Titanic project 
#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

#data visualizaiton
import seaborn as sns
import matplotlib.pyplot as plt

#importing the data
path = './project_practice/titanic'
df_train = pd.read_csv(path+'/train.csv')
df_test = pd.read_csv(path+'/test.csv')
combine = [df_train, df_test]

#Which features are avaible in the dataset
print(df_train.columns)
cat_var = ['Survived', 'Sex', 'Embraked', ]
ord_var = ['Pclass']
ncon_var = ['Age', 'Fare']
ndis_var = ['SibSp', 'Parch']

df_train.head(5) #preview the data

#detect the number of missing value
df_train.isnull().sum() # numbers of NAN in train by features
df_test.isnull().sum()
df_train.info() #info includes columns' name / dtype / memory usage
df_test.info()

#Distribution in numeric feature value
df_train.describe()
df_train.select_dtypes(include=['float64', 'int64']).plot(kind='box', subplots=True) 

df_train.describe(include=['O'])

#Analyze feature in pivot table 
pct = lambda x: x.count()/df_train.shape[0]
pivot_pclass = df_train[['Survived', 'Pclass']].groupby('Pclass').mean().sort_values(by='Survived', ascending=False)
pivot_sex = df_train[['Survived', 'Sex']].groupby('Sex').mean().sort_values(by='Survived', ascending=False)
pivot_sibsp = df_train[['Survived', 'SibSp']].groupby('SibSp').mean().sort_values(by='Survived', ascending=False)
pivot_parch = df_train[['Survived', 'Parch']].groupby('Parch').mean().sort_values(by='Survived', ascending=False)

#visualization 
#numeric side
hist_age = sns.FacetGrid(df_train, col='Survived')
hist_age.map(plt.hist, 'Age', bins=20)

#combine numeric and categorical side (Survived, Pclass, Age)
hist_pclass = sns.FacetGrid(df_train, col='Survived', row='Pclass')
hist_pclass.map(plt.hist, 'Age', bins=20)

#categorical side(Survived, Embarked, Sex, Pclass)
hist_embarked = sns.FacetGrid(df_train, row='Embarked', size=2.2, aspect=1.6)
hist_embarked.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='Set1')
hist_embarked.add_legend()

#combine numeric and categorical(Survived, Embarked, Sex, Fare)
hist_fare = sns.FacetGrid(df_train, row='Embarked', col='Survived')
hist_fare.map(sns.barplot, 'Sex', 'Fare', ci=None)

#Wrangling the data
#Drop the feature
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)

#Creating new feature from existing: Title
def extract_title(df):
    tmp = pd.DataFrame({'Title': df_train['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]})
    tmp.Title = tmp.Title.str.replace(' ', '')
    return tmp

df_train['Title'] = extract_title(df_train)
df_test['Title'] = extract_title(df_test)

pd.crosstab(df_train['Title'], df_train['Sex'])
df_train[['Title', 'Age']].groupby('Title').mean().round().sort_values(by='Age') #title band age accurately
df_train[['Title', 'Survived']].groupby('Title').mean().sort_values(by='Survived', ascending=False) #some title had higher survival rate
df_train.Title.unique()
##combine rare title to Rare 
def replace_title(df):
    df.Title = df.Title.replace(['Lady', 'theCountess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df.Title = df.Title.replace('Mme', 'Mrs')
    df.Title = df.Title.replace('Ms', 'Miss')
    df.Title = df.Title.replace('Mlle', 'Miss')
    return df

df_train = replace_title(df_train)
df_test = replace_title(df_test)

df_train[['Title', 'Survived']].groupby('Title').mean().sort_values(by='Survived', ascending=False)

##After extracting the Title, drop Name and PassengerId features
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name'], axis=1)

#Convert the categorical feature: Sex, Embarked, Title
##Sex 
df_train['Sex'] = df_train.Sex.map({'male': 0, 'female': 1}).astype(int)
df_test['Sex'] = df_test.Sex.map({'male': 0, 'female': 1}).astype(int)

##Embarked, transfer cat-embarked to ordinal
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode().loc[0])
df_test['Embarked'] = df_test['Embarked'].fillna(df_train['Embarked'].mode().loc[0]) #fill in the highest fequence port in train

df_train['Embarked'] = df_train['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).astype(int)
df_test['Embarked'] = df_test['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).astype(int)

##Title
df_train['Title'] = df_train.Title.map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
df_test['Title'] = df_test.Title.map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

#Filling missing value 
##Age 
def input_missing_age(df):
    age_tmp = np.zeros((2,3)) #creating a matrix to put our value
    for i in range(0, 2): #range for sex
        for j in range(0, 3): #range for pclass
            tmp_median = df[(df.Sex == i) & (df.Pclass == j+1)]['Age'].median()
            age_tmp[i, j] = tmp_median
    
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1) , 'Age'] = age_tmp[i, j] #return to Series and inputting the value  
    
    df['Age'] = df['Age'].astype(int)
    return df

df_train = input_missing_age(df_train)
df_test = input_missing_age(df_test)

##Age band

df_train['Ageband'] = pd.cut(df_train.Age, 5)
df_train[['Ageband', 'Survived']].groupby('Ageband', as_index=False).mean().sort_values(by='Ageband', ascending=True) #issue: by=Ageband without as_index could not sort

def age_band(df):
    df.loc[df.Age <= 16, 'Age'] = 0
    df.loc[(df.Age > 16) & (df.Age <= 32), 'Age'] = 1
    df.loc[(df.Age > 32) & (df.Age <= 48), 'Age'] = 2
    df.loc[(df.Age > 48) & (df.Age <= 64), 'Age'] = 3
    df.loc[df.Age > 64, 'Age'] = 4
    return df

df_train = age_band(df_train)
df_test = age_band(df_test) #use same interval 

df_train = df_train.drop('Ageband', axis=1)

##Fare 
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median()).round(2) #Only test data have missing fare value. Besides, we need to use a more sophisticated models to fill null value(like KNN)

##Fare band 
df_train['Fareband'] = pd.qcut(df_train.Fare, 4)
df_train.Fareband.value_counts()

def fare_band(df):
    df.loc[df.Fare <= 7.91, 'Fare'] = 0
    df.loc[(df.Fare > 7.91) & (df.Fare <= 14.454), 'Fare'] = 1
    df.loc[(df.Fare > 14.454) & (df.Fare <= 31.0), 'Fare'] = 2
    df.loc[df.Fare > 31, 'Fare'] = 3
    df.Fare = df.Fare.astype(int)
    return df

df_train = fare_band(df_train)
df_test = fare_band(df_test)

df_train = df_train.drop('Fareband', axis=1)

#Creating new Feature 
##Total family member = Sibsp + Parch
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # need to add the person himself
df_train[['FamilySize', 'Survived']].groupby('FamilySize').mean().sort_values(by='Survived', ascending=False)

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

##Is alone or not 
def isalone(df):
    df['IsAlone'] = 0
    df.loc[df.FamilySize == 1, 'IsAlone'] = 1 
    return df

df_train = isalone(df_train)
df_test = isalone(df_test)

df_train[['IsAlone', 'Survived']].groupby('IsAlone').mean()

df_train = df_train.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
df_test = df_test.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)

##Age + Pclass => need more explanation on this part 

##Build models - by 1st data preparation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, kernel_pca
from sklearn.pipeline import Pipeline
X = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

##print score function
def print_score(clf):
    print('Training score: {:.2f}'.format(clf.score(X_train, y_train)), '\n'
          'Test score: {:.2f}'.format(clf.score(X_test, y_test))) 

#Logistic 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=1)
lr.fit(X_train, y_train)
print_score(lr)

##Logistic with SCL and PCA
pipe_lr = Pipeline([('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print_score(pipe_lr)#decrease the accuracy score by implement SCL and PCA 

##SVM
from sklearn.svm import SVC
svc = SVC(random_state=1)
svc.fit(X_train, y_train)
print_score(svc)

#Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print_score(gnb)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print_score(dt)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
print_score(rf)

#Grandient boost
from sklearn.ensemble import GradientBoostingClassifier
gdc = GradientBoostingClassifier()
gdc.fit(X_train, y_train)
print_score(gdc)

#Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
print_score(perceptron)


## Next step: Using k-fold validation to evaluate the average performace of each models
scores = cross_val_score(estimator=lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print ('CV accuracy: {0:.2f} +/- {1:.2f}'.format(np.mean(scores), np.std(scores)))


