

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data= pd.read_csv("D:\\excelR\\Data science notes\\Decision Tree\\asgmnt\\Fraud_check.csv")
data.head()

data['marital_status'].unique()
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data['marital_status']=label_encoder.fit_transform (data['marital_status'])
data
data['marital_status'].unique()
data['tax_income'].unique()
data.tax_income.value_counts



labels=["risky","good"]
bins=[0,30000,100000]
data1=pd.cut(data.iloc[:,2],labels=labels,bins=bins)
data.drop(["tax_income"], axis=1,inplace=True)

data=pd.concat([data,data1],axis=1)
data
data.shape
colnames = list(data.columns)

predictors = colnames[:5]
predictors
target = colnames[5]
target


from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)


from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])


preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
np.mean(train.tax_income == model.predict(train[predictors])) ## 100

# Accuracy = Test
np.mean(preds==test.tax_income)  ## 71.67
