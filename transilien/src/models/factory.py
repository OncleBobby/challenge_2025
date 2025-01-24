from importlib import import_module

class ModelFactory():
  def __init__(self, configurations, x_train, y_train, x_test, y_test):
    self.configurations = configurations
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
  def get_model(self, name):
    configuration = self.configurations[name]
    class_str = configuration['class']
    params = configuration['params'] if 'params' in configuration else None
    module_path, class_name = class_str.rsplit('.', 1)
    module = import_module(module_path)
    model = getattr(module, class_name)(self.x_train, self.y_train, self.x_test, self.y_test, params)
    model.name = name
    return model
  def get_models(self):
    models = []
    for name, configuration in self.configurations.items():
      models.append(self.get_model(name))
    return models