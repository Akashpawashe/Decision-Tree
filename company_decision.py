

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:12:13 2020

@author: Ketan
"""


import pandas as pd
import numpy as np

company = pd.read_csv("D:\\excelR\\Data science notes\\Decision Tree\\asgmnt\\Company_Data.csv")
company.head()
company.columns

#converting into categorical data
np.median(company["Sales"]) #7.49
company["sales"]="<=7.49"
company.loc[company["Sales"]>=7.49,"sales"]=">=7.49"

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
company["ShelveLoc"]=lb.fit_transform(company["ShelveLoc"])
company["Urban"]=lb.fit_transform(company["Urban"])
company["US"]=lb.fit_transform(company["US"])
company["sales"]=lb.fit_transform(company["sales"])

company.drop(["Sales"],inplace=True,axis=1)

colnames = list(company.columns)
predictors = colnames[:10]
target = colnames[10]

X = company[predictors]
Y = company[target]

from sklearn.model_selection import train_test_split
train,test = train_test_split(company,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,1:11],train['sales'])

np.mean(train['sales'] == model.predict(train.iloc[:,1:11])) ## 1

np.mean(test['sales'] == model.predict(test.iloc[:,1:11])) ## 1

