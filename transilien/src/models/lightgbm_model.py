from .model import Model

class LightgbmModel(Model):
  def __init__(self, x_train, y_train, x_test, y_test, params=None):
    Model.__init__(self, x_train, y_train, x_test, y_test, params)
    self.name = 'catboost'
    self.model = None
  def fit(self):
    from lightgbm import LGBMRegressor
    self.model = LGBMRegressor(verbosity=-1)
    self.model.fit(self.x_train, self.y_train)
    return self.model