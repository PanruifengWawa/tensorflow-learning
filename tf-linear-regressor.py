## y = x1 + 3x2 + 4
import tensorflow as tf
import numpy as np


#定义x1和x2
feature_columns = [tf.feature_column.numeric_column("x1"),tf.feature_column.numeric_column("x2")]


#优化器，可选，参数得慎重
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

#如果没有优化器，把优化器参数去掉
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer,
    model_dir="linear_model/"
)


#设定输入
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x1": np.array([3,2,4]), "x2": np.array([1,2,5])},
      y=np.array([10,12,23]),
      num_epochs=None,
      shuffle=True)

#训练模型
linear_regressor.train(input_fn=train_input_fn, steps=10000)


#设定预测输入
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x1": np.array([3]),"x2":np.array([0])},
      num_epochs=1,
      shuffle=False)

#预测结果
predictions = list(linear_regressor.predict(input_fn=predict_input_fn))
print(predictions)
