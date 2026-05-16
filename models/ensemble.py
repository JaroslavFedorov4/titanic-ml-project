import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from catboost import Pool

def average_pred(full_results, X, y, categorical_features):
    all_preds = []
    X_cat = X.copy()
    for item in full_results:
        model_name = item["model"]
        model = item["best_estimator"]
        if model_name == "Catboost":
            X_cat[categorical_features] = X_cat[categorical_features].astype(str)
            pred = model.predict_proba(X_cat)[:,1]
            all_preds.append(pred)
        else:
            pred = model.predict_proba(X)[:,1]
            all_preds.append(pred)

    pred_avg = np.mean(all_preds, axis=0)

    final_pred = (pred_avg >= 0.5).astype(int)

    acc = accuracy_score(y, final_pred)

    return acc

def voting_pred(full_results, X, y, categorical_features):
    all_preds = []
    X_cat = X.copy()
    for item in full_results:
        model_name = item["model"]
        model = item["best_estimator"]
        if model_name == "Catboost":
            X_cat[categorical_features] = X_cat[categorical_features].astype(str)
            pred = model.predict(X_cat)
            all_preds.append(pred)
        else:
            pred = model.predict(X)
            all_preds.append(pred)

    pred_avg = np.mean(all_preds, axis=0)

    final_pred = (pred_avg >= 0.5).astype(int)

    acc = accuracy_score(y, final_pred)

    return acc

def stacking_acc(full_results, X_train, y_train, X_val, y_val, categorical_features, n_splits, random_state):
    n_samples = len(X_train)
    n_models = len(full_results)

    meta_X_train = np.zeros((n_samples, n_models))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for model_idx, item in enumerate(full_results):
        model_name = item["model"]
        model_base = item["best_estimator"]

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx].copy()
            X_fold_val = X_train.iloc[val_idx].copy()

            y_fold_train = y_train.iloc[train_idx].copy()

            model = clone(model_base)

            if model_name == "Catboost":
                X_fold_train[categorical_features] = X_fold_train[categorical_features].fillna("Unknown").astype(str)
                X_fold_val[categorical_features] = X_fold_val[categorical_features].fillna("Unknown").astype(str)

                train_pool = Pool(
                    data=X_fold_train,
                    label=y_fold_train,
                    cat_features=categorical_features
                )

                model.fit(train_pool, verbose=0)

                pred = model.predict_proba(X_fold_val)[:,1]

            else:
                model.fit(X_fold_train, y_fold_train)
                pred = model.predict_proba(X_fold_val)[:,1]

            meta_X_train[val_idx, model_idx] = pred

    meta_model = LogisticRegression(max_iter=1000)

    meta_model.fit(meta_X_train, y_train)

    meta_X_val = np.zeros((len(X_val), n_models))

    for model_idx, item in enumerate(full_results):
        model_name = item["model"]
        model = item["best_estimator"]

        if model_name == "Catboost":
            X_val_cat = X_val.copy()
            X_val_cat[categorical_features] = X_val_cat[categorical_features].fillna("Unknown").astype(str)

            pred = model.predict_proba(X_val_cat)[:,1]

        else:
            pred = model.predict_proba(X_val)[:,1]

        meta_X_val[:, model_idx] = pred

    final_pred = meta_model.predict(meta_X_val)

    acc = accuracy_score(y_val, final_pred)

    return acc