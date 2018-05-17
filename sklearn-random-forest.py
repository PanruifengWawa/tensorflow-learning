
# coding: utf-8

# In[107]:


#labelencoder
from sklearn.preprocessing import LabelEncoder  
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def read_data(path):
    data = pd.read_csv(path,sep=",")
    return data

def label_encode(data):
    labelencoder = LabelEncoder()  
    for col in data.columns:  
        data[col] = labelencoder.fit_transform(data[col])  
        print(col,labelencoder.classes_)
    return data

all_data = read_data("car_data/alldata.txt")
all_data_feature = label_encode(all_data.drop(["classes"], axis=1))

train = all_data_feature[0:1436]
test = all_data_feature[1436:1728]
train_target = np.array(all_data["classes"][0:1436])
test_target = np.array(all_data["classes"][1436:1728])


rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(train,train_target)

score = rnd_clf.score(test,test_target)
print(score)
predict = rnd_clf.predict(test)
print(predict)


# In[115]:


#特征二值化
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#是否使用稀疏矩阵
my_sparse = False

def read_data(path):
    data = pd.read_csv(path,sep=",")
    return data

def vectorizer(data):
    
    vec = DictVectorizer(sparse=my_sparse)
    data= vec.fit_transform(data.to_dict(orient='record'))
    print(vec.get_feature_names())
    return data

all_data = read_data("car_data/alldata.txt")
all_data_feature = vectorizer(all_data.drop(["classes"], axis=1))


train = all_data_feature[0:1436]
test = all_data_feature[1436:1728]
train_target = np.array(all_data["classes"][0:1436])
test_target = np.array(all_data["classes"][1436:1728])



rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(train,train_target)


score = rnd_clf.score(test,test_target)
print(score)
predict = rnd_clf.predict(test)
print(predict)

