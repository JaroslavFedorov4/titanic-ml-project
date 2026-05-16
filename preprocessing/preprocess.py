from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def prep(numeric_features, categorical_features):
    normalize_numeric = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    normalize_categorical = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', normalize_numeric , numeric_features),
        ('categorical', normalize_categorical, categorical_features)
    ])

    return preprocessor

def prep_NN(numeric_features, categorical_features):
    normalize_numeric = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    normalize_categorical = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor_num = ColumnTransformer([
        ('num', normalize_numeric, numeric_features)
    ])

    preprocessor_cat = ColumnTransformer([
        ('cat', normalize_categorical, categorical_features)
    ])

    return preprocessor_num, preprocessor_cat