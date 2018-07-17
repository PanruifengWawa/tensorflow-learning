#数据如
#sentence classes
#你们公司在哪里 AT

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

labels=["AT","CI","PI","WE","NN"]
def load_data(filepath):
    data = pd.read_csv(filepath,sep=" ")
    target = pd.Series([labels.index(val) for val in data["classes"]])
    return data,target


embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
    model_dir="text_model/")


#train
train_data ,target = load_data("trainData/trainData.txt")
train_data = pd.DataFrame({"sentence":train_data["sentence"]})

train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_data, y=target, num_epochs=None, shuffle=True)

estimator.train(input_fn=train_input_fn, steps=1000)


# read test data
test_data, test_target = load_data("trainData/trainSet_GY.txt")
test_input_fn = tf.estimator.inputs.pandas_input_fn(
     x=test_data, y=test_target, num_epochs=1, shuffle=False)
#test accuray
test_accuray = estimator.evaluate(input_fn=test_input_fn)
print(test_accuray)
#test result
test_result = estimator.predict(input_fn=test_input_fn)
class_index = [x["class_ids"][0] for x in test_result]
result = [labels[x] for x in class_index]
x_target = [labels[x] for x in test_target]
result = pd.DataFrame({"sentence": test_data["sentence"],"classes":x_target,"predict":result})
print(result)

#predict
my_sentence = pd.Series(["请问你们公司的地点在哪里"])
prediction_data = pd.DataFrame({"sentence": my_sentence})

predict_input_fn = tf.estimator.inputs.pandas_input_fn(
     x=prediction_data, num_epochs=1, shuffle=False)
prediction = estimator.predict(input_fn=predict_input_fn)
b = [x["class_ids"][0] for x in prediction]
print(my_sentence[0],labels[b[0]])
