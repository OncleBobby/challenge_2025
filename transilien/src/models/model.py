import pandas, logging, time, numpy

class Model:
  def __init__(self, x_train, y_train, x_test, y_test, params=None):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.params = params
    self.name = self.__class__.__name__
    self.model = None
  def fit(self):
    pass
  def predict(self, x):
    if self.model is None:
       self.fit()
    return pandas.DataFrame(self.model.predict(x)).rename(columns={0: 'p0q0'})
  def evaluate(self):
      start = time.time()
      y_pred = self.predict(self.x_test)
      from sklearn.metrics import mean_absolute_error
      mae = mean_absolute_error(self.y_test, y_pred)
      end = time.time()
      logging.info(f'{self.name}={numpy.round(mae, 4)} in {numpy.round((end-start), 2)}s')
      return mae
  def save(self, X, file_name_pattern='../data/test/y_predict{name}.csv'):
      y_pred = self.predict(X)
      y_pred.to_csv(file_name_pattern.format(name=self.name), index=False)
