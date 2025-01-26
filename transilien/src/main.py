import logging, pandas, yaml, logging.config, time, numpy

def prepare_data(x_columns, train_path = f'../data/train/', test_path = f'../data/test/'):
    from sklearn.preprocessing import LabelEncoder
    x_category_columns = ['train', 'gare', 'date']
    y_column = 'p0q0'
    x_train = pandas.read_csv(f'{train_path}x_train_final.csv', index_col=False)
    y_train = pandas.read_csv(f'{train_path}y_train_final.csv', index_col=False)

    x_to_predict = pandas.read_csv(f'{test_path}x_test_final.csv', index_col=False)[x_columns]

    encoder = LabelEncoder() 
    for column in x_category_columns:
        if column in x_to_predict.columns:
            x_train[column]= encoder.fit_transform(x_train[column])
            x_to_predict[column] = encoder.transform(x_to_predict[column])

    def _train_test_split(x_train, y_train, test_size):
        from random import shuffle
        dates = x_train['date'].drop_duplicates().to_list()
        i = int(round(len(dates)*test_size, 0))
        shuffle(dates)
        train_dates = dates[:i]
        test_dates = dates[i:]
        return \
            x_train[x_train['date'].isin(train_dates)], \
            x_train[x_train['date'].isin(test_dates)], \
            y_train[x_train['date'].isin(train_dates)], \
            y_train[x_train['date'].isin(test_dates)]

    # from sklearn.model_selection import train_test_split
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    x_train, x_test, y_train, y_test = _train_test_split(x_train, y_train, test_size=0.2)
    return x_train[x_columns], x_test[x_columns], y_train[[y_column]], y_test[[y_column]], x_to_predict[x_columns]

def evaluate_models(x_train, y_train, x_test, y_test):
    from models.factory import ModelFactory
    with open('../conf/models.yml', 'r') as file:
        configurations = yaml.safe_load(file)
    factory = ModelFactory(configurations, x_train, y_train, x_test, y_test)
    lines = []
    for model in factory.get_models():
        start = time.time()
        score = model.evaluate()
        end = time.time()
        duration = numpy.round((end-start), 2)
        logging.info(f'{model.name}={numpy.round(score, 4)} in {duration}s')
        lines.append({'name': model.name, 'score': score, 'duration': duration})
    df = pandas.DataFrame(lines).sort_values(by=['score'], ascending=True)
    return df

if __name__ == "__main__":
    from models.factory import ModelFactory
    import yaml
    train_path = f'transilien/data/train/'
    test_path = f'transilien/data/test/'
    with open('transilien/conf/log.yml', 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    with open('transilien/conf/models.yml', 'r') as file:
        configurations = yaml.safe_load(file)
    x_columns = ['train', 'gare', 'date', 'arret', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
    x_columns = ['gare', 'arret', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
    x_train, x_test, y_train, y_test, x_to_predict = prepare_data(x_columns, train_path, test_path)
    factory = ModelFactory(configurations, x_train, y_train, x_test, y_test)
    for model in factory.get_models():
        evaluation = model.evaluate()