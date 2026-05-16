from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import torch

TARGET = "Survived"

NUMERIC_FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

CATEGORICAL_FEATURES = ['Sex', 'Cabin', 'Embarked']

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

RANDOM_STATE = 42
N_SPLITS = 3
SCORING = "accuracy"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(algorithm="brute", n_jobs=1),
    "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "XGBClassifier": XGBClassifier(random_state=RANDOM_STATE),
    "LGBMClassifier": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, force_col_wise=True, verbosity=-1),
    "Catboost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=100)
}

param_grids = {
    "LogisticRegression": {
        'model__penalty': ['elasticnet'],
        'model__solver': ['saga'],
        'model__C': [0.01, 0.1, 1],
        'model__l1_ratio': [0.2, 0.5, 0.8]
    },
    "KNN": {
        'model__n_neighbors': [3],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    },
    "RandomForestClassifier": {
        'model__n_estimators': [64, 100],
        'model__max_features': [2, 3, 4],
        'model__max_depth': [3, 5, 7, 9],
        'model__bootstrap': [True, False]
    },
    "DecisionTreeClassifier": {
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [3, 4, 5],
        'model__min_samples_leaf': [1, 2, 3, 4, 5],
        'model__criterion': ['gini', 'entropy']
    },
    "XGBClassifier": {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 4],
        'model__subsample': [0.8, 0.6, 1.0],
        'model__colsample_bytree': [0.8, 0.6, 1.0]
    },
    "Catboost": {
        "iterations": [100, 200],
        "learning_rate": [0.05, 0.1],
        "depth": [3, 4],
        "early_stopping_rounds": [50, 100, 150],
        "l2_leaf_reg": [3, 5, 7]
    },
    "LGBMClassifier": {
        "model__n_estimators": [150],
        "model__learning_rate": [0.05],
        "model__num_leaves": [15],
        "model__max_depth": [3],
        "model__verbosity": [-1]
    }
}