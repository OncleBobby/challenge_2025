import logging, pandas

def prepare_data(x_columns):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split


    train_path = f'../data/train/'
    test_path = f'../data/test/'
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
def get_model(x_train, y_train):
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.fit(x_train, y_train)
    return model
def evaluate(model, x_test, y_test):
    from sklearn.metrics import mean_absolute_error
    y_pred = pandas.DataFrame(model.predict(x_test))
    return mean_absolute_error(y_test, y_pred)
