import numpy as np
import pandas as pd

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_new_features(df):
    df = df.copy()
    categorical_features = CATEGORICAL_FEATURES.copy()
    numeric_features = NUMERIC_FEATURES.copy()

    df["StrCabin"] = df["Cabin"].str[0].fillna("Unknown")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Sex_Pclass"] = (df["Sex"].astype(str) + "_" + df["Pclass"].astype(str))

    categorical_features.extend(["StrCabin", "Sex_Pclass"])

    numeric_features.extend(["FamilySize", "IsAlone"])

    df.drop(["Cabin", "SibSp", "Parch"], axis=1, inplace=True)

    numeric_features = [
        col for col in numeric_features
        if col in df.columns
    ]

    categorical_features = [
        col for col in categorical_features
        if col in df.columns
    ]

    features = numeric_features + categorical_features

    return df[features], numeric_features, categorical_features
