from .model import Model
import keras, logging

class KerasModel(Model):
  def __init__(self, x_train, y_train, x_test, y_test, params={}):
    Model.__init__(self, x_train, y_train, x_test, y_test, params)
    self.name = 'keras'
    self.params = params
    self.model = None
    self.history = None
  def fit(self):
    layers = [keras.Input(shape=(self.x_train.shape[1],))]
    for layer in self.params['layers']:
      layers.append(keras.layers.Dense(layer['units'], activation=layer['activation'],  kernel_initializer='normal'))
    # layers.append(keras.layers.Dense(1, activation="linear"))
    self.model = keras.Sequential(layers)
    batch_size = self.params['batch_size'] if 'batch_size' in self.params else 16
    epochs = self.params['epochs'] if 'epochs' in self.params else 10
    validation_split = self.params['validation_split'] if 'validation_split' in self.params else 16
    optimizer = self.params['optimizer'] if 'optimizer' in self.params else 'adam'
    self.model.compile(loss='mean_squared_error', optimizer=optimizer)
    self.history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return self.model