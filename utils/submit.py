import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from preprocessing.preprocess import prep
from sklearn.pipeline import Pipeline
from config import RANDOM_STATE

from preprocessing.feature_engineering import make_new_features

def make_submission(X, y, best_params, numeric_features, categorical_features):
    test_df = pd.read_csv("data/test.csv")

    test_id = test_df["PassengerId"]

    X_test, _, _ = make_new_features(test_df)

    clean_params = {}

    for key, value in best_params.items():
        clean_key = key.replace("model__", "")
        clean_params[clean_key] = value

    pipeline = Pipeline([
        ("preprocessor", prep(numeric_features, categorical_features)),
        ("model", XGBClassifier(**clean_params, random_state=RANDOM_STATE))
    ])

    pipeline.fit(X, y)

    preds = pipeline.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": preds
    })

    submission.to_csv('data/submission.csv', index=False)

    print("submission.csv saved!")