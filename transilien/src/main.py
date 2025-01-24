import logging, pandas, logging.config

def prepare_data(x_columns, train_path = f'../data/train/', test_path = f'../data/test/'):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    x_category_columns = ['train', 'gare', 'date']

    y_column = 'p0q0'
    x_train = pandas.read_csv(f'{train_path}x_train_final.csv', index_col=False)[x_columns]
    y_train = pandas.read_csv(f'{train_path}y_train_final.csv', index_col=False)[[y_column]]

    x_to_predict = pandas.read_csv(f'{test_path}x_test_final.csv', index_col=False)[x_columns]

    encoder = LabelEncoder() 
    for column in x_category_columns:
        if column in x_train.columns:
            x_train[column]= encoder.fit_transform(x_train[column])
            x_to_predict[column] = encoder.transform(x_to_predict[column])

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test, x_to_predict


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