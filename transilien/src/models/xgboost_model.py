from .model import Model

class XgboostModel(Model):
  def __init__(self, x_train, y_train, x_test, y_test, params=None):
    Model.__init__(self, x_train, y_train, x_test, y_test, params)
    self.name = 'xgboost'
    self.model = None
  def fit(self):
    from xgboost import XGBRegressor
    self.model = XGBRegressor()
    self.model.fit(self.x_train, self.y_train)
    return self.model