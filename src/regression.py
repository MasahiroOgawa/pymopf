from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

# prepare data
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# create model
model = keras.models.Sequential([keras.layers.Dense(20, activation="relu", input_shape=X_train.shape[1:]),
                                 keras.layers.Dense(1)])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))

# train
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid,y_valid))

# test
model.evaluate(X_test, y_test)
plt.plot(pd.DataFrame(history.history))
plt.show()