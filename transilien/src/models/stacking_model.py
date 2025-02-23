from .model import Model
from importlib import import_module
from sklearn.ensemble import StackingRegressor

class StackingModel(Model):
  def __init__(self, x_train, y_train, x_test, y_test, params=None):
    Model.__init__(self, x_train, y_train, x_test, y_test, params)
    self.name = 'stacking'
    self.model = None
  def fit(self):
    estimators=[(class_name, self._get_intance(class_name)) for class_name in self.params['class_names']]
    final_estimator = self._get_intance(self.params['final_estimator'])
    self.model = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
    self.model.fit(self.x_train, self.y_train)

  def _get_intance(self, class_str):
      s = class_str.split('.')
      module_path = '.'.join(s[:-1])
      class_name = s[-1]
      module = import_module(module_path)
      params = {}
      # params = self.params.copy()
      # params.pop('class_name')
      estimator = getattr(module, class_name)(**params)
      return estimator