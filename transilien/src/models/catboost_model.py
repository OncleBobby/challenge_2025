from .model import Model

class CatboostModel(Model):
  def __init__(self, x_train, y_train, x_test, y_test, params=None):
    Model.__init__(self, x_train, y_train, x_test, y_test, params)
    self.name = 'xgboost'
    self.model = None
  def fit(self):
    from catboost import CatBoostRegressor
    self.model = CatBoostRegressor()
    self.model.fit(self.x_train, self.y_train, verbose=False)
    return self.model