from __future__ import absolute_import
from __future__ import division


import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
            f.write(raw)

  # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("a"),tf.feature_column.numeric_column("b"),tf.feature_column.numeric_column("c"),tf.feature_column.numeric_column("d")]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="iris_model/")
  # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"a": training_set.data.T[0],"b": training_set.data.T[1],"c": training_set.data.T[2],"d": training_set.data.T[3]},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
    classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
       x={"a": test_set.data.T[0],"b": test_set.data.T[1],"c": test_set.data.T[2],"d": test_set.data.T[3]},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)
   
  # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"a": np.array([5.4,6.4]),"b":np.array([3.9,2.8]),"c":np.array([1.3,5.6]),"d":np.array([0.4,2.1])},
      num_epochs=1,
      shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
    main()