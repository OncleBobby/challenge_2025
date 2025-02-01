import logging, pandas, yaml, logging.config, time, numpy

def encode_with_label(x_train, x_valid, x_category_columns):
    from sklearn.preprocessing import LabelEncoder
    for column in x_category_columns:
        if column in x_valid.columns:
            encoder = LabelEncoder() 
            x_train[column]= encoder.fit_transform(x_train[column])
            x_valid[column] = encoder.transform(x_valid[column])
    return x_train, x_valid
def encode_with_one_hot(x_train, x_valid, x_category_columns):
    from sklearn.preprocessing import OneHotEncoder
    columns = ['gare']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cols_train = pandas.DataFrame(encoder.fit_transform(x_train[columns]))
    cols_valid = pandas.DataFrame(encoder.transform(x_valid[columns]))

    # One-hot encoding removed index; put it back
    cols_train.index = x_train.index
    cols_valid.index = x_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    numeric_x_train = x_train.drop(columns, axis=1)
    numeric_x_valid = x_valid.drop(columns, axis=1)

    # Add one-hot encoded columns to numerical features
    new_x_train = pandas.concat([numeric_x_train, cols_train], axis=1)
    new_x_valid = pandas.concat([numeric_x_valid, cols_valid], axis=1)
    return new_x_train, new_x_valid
def prepare_data(x_columns, train_path = f'../data/train/', test_path = f'../data/test/'):
    from sklearn.preprocessing import LabelEncoder
    x_category_columns = ['train', 'gare', 'date']
    y_column = 'p0q0'
    x_news_columns = x_columns.copy()
    x_news_columns.append('date')
    x_train = pandas.read_csv(f'{train_path}x_train_final.csv', index_col=False)[x_news_columns]
    y_train = pandas.read_csv(f'{train_path}y_train_final.csv', index_col=False)[[y_column]]
    x_valid = pandas.read_csv(f'{test_path}x_test_final.csv', index_col=False)[x_news_columns]

    # x_train, x_valid = encode_with_label(x_train, x_valid, x_category_columns)
    x_train, x_valid = encode_with_one_hot(x_train, x_valid, x_category_columns)

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
    x_train = x_train.drop('date', axis=1)
    x_test = x_test.drop('date', axis=1)
    x_valid = x_valid.drop('date', axis=1)
    return x_train, x_test, y_train, y_test, x_valid

def evaluate_models(x_train, y_train, x_test, y_test, x_valid=None):
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
        if not x_valid is None:
            model.save(x_valid, file_name_pattern='../data/test/y_predict_{name}.csv')
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
    x_train, x_test, y_train, y_test, x_valid = prepare_data(x_columns, train_path, test_path)
    factory = ModelFactory(configurations, x_train, y_train, x_test, y_test)
    for model in factory.get_models():
        evaluation = model.evaluate()