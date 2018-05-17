
# coding: utf-8

# In[11]:


#LogisticRegressor 
import pandas as pd
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression

def read_data(path):
    data = pd.read_csv(path,sep=",")
    return data



data = read_data("iris_data/train.txt")
X_train = data.drop(["label"], axis=1)
y_train= data["label"]

classifier=LogisticRegression()
classifier.fit(X_train,y_train)

X_test=[[5.1,3.5],[6.4,3.1]]
predictions=classifier.predict(X_test)
print(predictions)

