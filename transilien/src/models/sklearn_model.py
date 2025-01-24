from .model import Model
from importlib import import_module

class SklearnModel(Model):
  def __init__(self, x_train, y_train, x_test, y_test, params=None):
    Model.__init__(self, x_train, y_train, x_test, y_test, params)
    self.name = 'sklearn'
    self.model = None
  def fit(self):
    class_str = self.params['class_name']
    s = class_str.split('.')
    module_path = '.'.join(s[:-1])
    class_name = s[-1]
    module = import_module(module_path)
    params = self.params.copy()
    params.pop('class_name')
    self.model = getattr(module, class_name)(**params)
    self.model.fit(self.x_train, self.y_train)