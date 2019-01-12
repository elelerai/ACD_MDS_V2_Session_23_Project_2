# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:26:13 2019

@author: Eliud Lelerai
"""

import sqlite3
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt




cnx = sqlite3.connect('database.sqlite')

# Importing the dataset

df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


df.head()
df.columns
print(df.keys())
df.describe()
df.info()

# Exploring data: Cleaning and Transforming Data

df.preferred_foot.value_counts(dropna=False)
df['preferred_foot'].replace(['right','left'],[1,2],inplace=True)
df['preferred_foot'] = pd.to_numeric(df['preferred_foot'],errors='coerce')


df.attacking_work_rate.value_counts(dropna=False)
df['attacking_work_rate'].replace(['low','medium','high'],[1,2,3],inplace=True)
df['attacking_work_rate'] = pd.to_numeric(df['attacking_work_rate'],errors='coerce')


df.defensive_work_rate.value_counts(dropna=False)
df['defensive_work_rate'].replace(['low','medium','high'],[1,2,3],inplace=True)
df['defensive_work_rate'] = pd.to_numeric(df['defensive_work_rate'],errors='coerce')

df = df.drop_duplicates()
df=df.dropna(how='any')

# Getting the player rating (y) and player attributes (X)

X=df.iloc[:,5:42].values
y=df.iloc[:,4].values

# Splitting thr data to train and tests data sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.30, random_state = 0)



# Fitting the model 
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

model=lm.fit(X_train,y_train)

#print the coefficients

print(model.coef_)



# Trying to predict the y ( player rating) for the test data-features(independent variables(X_test)
 
predictions=lm.predict(X_test)
 
# Accuracy of the prediction

confidence = lm.score(X_test, y_test)
print("The predictions are:",predictions)
print("The confidence is:",confidence)

import matplotlib.pyplot as plt

plt.scatter(y_test,predictions)




#Check the R2- the proportion of variance in the dependent variable explained by the predictors


