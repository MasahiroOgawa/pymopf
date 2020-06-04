import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# prepare data
(X_train_full, y_train_full),(X_test, y_test) = keras.datasets.fashion_mnist.load_data()
num_valid = 5000
X_valid = X_train_full[:num_valid] / 255.
X_train = X_train_full[num_valid:] / 255.
y_valid = y_train_full[:num_valid]
y_train = y_train_full[num_valid:]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# check data
X_train_full.shape
X_train_full.dtype
plt.imshow(X_train_full[0],cmap="binary")
plt.axis('off')

# create model and train
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28,28]),
                                 keras.layers.Dense(200, activation="relu"),
                                 keras.layers.Dense(50),
                                 keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid,y_valid))

# check result
pd.DataFrame(history.history).plot()
plt.show()
