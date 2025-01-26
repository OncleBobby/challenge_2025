from .model import Model
import keras, pandas

class KerasModel(Model):
  def __init__(self, x_train, y_train, x_test, y_test, params=None):
    Model.__init__(self, x_train, y_train, x_test, y_test, params)
    self.name = 'catboost'
    self.model = None
  def fit(self):
    layers = [keras.Input(shape=(self.x_train.shape[1],))]
    layers.append(keras.layers.Dense(10, activation="relu"))
    layers.append(keras.layers.Dense(10, activation="relu"))
    layers.append(keras.layers.Dense(1, activation="linear"))
    self.model = keras.Sequential(layers)
    # self.model.summary()
    self.model.compile(loss='mean_squared_error', optimizer='adam')
    history = self.model.fit(self.x_train, self.y_train, batch_size=16, epochs=10, validation_split=0.3)
    return self.model