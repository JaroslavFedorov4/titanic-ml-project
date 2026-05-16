import pandas as pd 

from config import TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from preprocessing.feature_engineering import make_new_features

def load_data(path):
    df = pd.read_csv(path)

    X, numeric_features, categorical_features = make_new_features(df)
    y = df[TARGET]

    return X, y, numeric_features, categorical_features