import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import warnings
from statsmodels import robust
from sklearn.model_selection import KFold, train_test_split
from exp.mappings import alg_map, p_dist_map
from exp.features import load_train_features, load_test_features, feature_sel_pc
import copy
warnings.filterwarnings("ignore")

def train_get_test_preds(X, Y, X_test, params, alg="lr"):

    # retrieve algorithm
    model_cls, model_type = alg_map[alg]

    if model_type == 'sklearn':
        model = model_cls(**params)
        model.fit(X, Y)
        y_pred = model.predict(X_test)
    if model_type == 'lgb':
        model = model_cls(**params)
        model.fit(X, Y)
        y_pred = model.predict(X_test)
    if model_type == 'xgb':
        # add early stopping option
        train_data = model_cls.DMatrix(data=X, label=Y, feature_names=X.columns)
        model = model_cls.train(dtrain=train_data, num_boost_round=num_boost_round, params=params)
        y_pred = model.predict(model_cls.DMatrix(X_test, feature_names=X.columns))
    if model_type == 'cat':
        model = model_cls(**params)
        model.fit(X, Y, cat_features=[], verbose=False)
        y_pred = model.predict(X_test)

    y_pred = y_pred.reshape(-1, )
    return model, y_pred


def generate_train_val(X, nearest_train, sample_prop=.10, rs=None):
    # sample with replacement from nearest train segments to test set
    val_idxs = list(nearest_train.sample(n=int(sample_prop*len(X.index)), replace=True, random_state=rs))
    no_repeats_val_idxs = set(val_idxs)
    train_idxs = [i for i in range(len(X.index)) if i not in no_repeats_val_idxs]
    return train_idxs, val_idxs


def generate_cv_indices(X, Y, X_test, n_fold=10, test_val=False, dist="l1", top_k_features=None,
                        sample_prop=.10, shuffle=True, rs=None):
        if test_val:
            dist_func = p_dist_map[dist]
            X, X_test = X.copy(), X_test.copy()
            X, X_test = feature_sel_pc(X, Y, X_test, top_k=top_k_features)
            dist_train_test = pd.DataFrame(dist_func(X, X_test))
            # minimum distance training segment per test segment
            nearest_train = dist_train_test.idxmin(axis=0)
            for i in range(n_fold):
                yield generate_train_val(X, nearest_train, sample_prop=sample_prop, rs=rs+i)
        else:
            folds = KFold(n_splits=n_fold, shuffle=shuffle, random_state=rs)
            for inds in folds.split(X):
                yield inds


def train_model(params, fs="standard_scaled", n_fold=10, test_val = False, dist="l1", top_k_features=None,
                sample_prop=.10, shuffle=True, rs=None, alg="lr", plot_feature_importance=False, test_eval=False,
                X=None, Y=None, X_test=None):
    """Taken from the `Earthquakes FE. More features and samples` kaggle notebook"""

    if fs:
        X, Y = load_train_features(fs)
        if test_eval:
            X_test = load_test_features(fs)

    params = copy.deepcopy(params)
    if n_fold is None:
        return train_get_test_preds(X, Y, X_test, params, alg)

    # retrieve algorithm
    model_cls, model_type = alg_map[alg]

    oof = np.zeros(len(X.index))
    if X_test is not None:
        prediction = np.zeros(len(X_test.index))
    scores = []
    feature_importance = pd.DataFrame()

    early_stopping = params.pop("early_stopping", {})
    num_boost_round = params.pop("num_boost_round", 10)

    folds_inds = generate_cv_indices(X, Y, X_test, n_fold=n_fold, test_val=test_val, dist=dist,
                                     top_k_features=top_k_features, sample_prop=sample_prop, shuffle=shuffle, rs=rs)
    for fold_n, (train_index, valid_index) in enumerate(folds_inds):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]

        if model_type == 'sklearn':
            model = model_cls(**params)
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid)

            if X_test is not None:
                y_pred = model.predict(X_test)
        if model_type == 'lgb':
            model = model_cls(**params)
            if early_stopping:
                test_size = early_stopping.get("test_size", .10)
                X_t_train, X_e_train, y_t_train, y_e_train = train_test_split(X_train, y_train, test_size = test_size)
                eval_metric = early_stopping.get("eval_metric", "mae")
                early_stopping_rounds = early_stopping.get("early_stopping_rounds", 200)
                model.fit(X_t_train, y_t_train, eval_set=[(X_e_train, y_e_train)], eval_metric=eval_metric,
                          verbose=10000, early_stopping_rounds=early_stopping_rounds)
            else:
                model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid)
            if X_test is not None:
                y_pred = model.predict(X_test)

        if model_type == 'xgb':
            if early_stopping:
                test_size = early_stopping.get("test_size", .10)
                early_stopping_rounds = early_stopping.get("early_stopping_rounds", "200")
                X_t_train, X_e_train, y_t_train, y_e_train = train_test_split(X_train, y_train, test_size = test_size)
                train_t_data = model_cls.DMatrix(data=X_t_train, label=y_t_train, feature_names=X_t_train.columns)
                valid_e_data = model_cls.DMatrix(data=X_e_train, label=y_e_train, feature_names=X_e_train.columns)
                watchlist = [(train_t_data, 'train_t'), (valid_e_data, 'valid_e')]

                model = model_cls.train(dtrain=train_t_data, num_boost_round=num_boost_round, evals=watchlist,
                                        early_stopping_rounds=early_stopping_rounds, verbose_eval=500, params=params)
                y_pred_valid = model.predict(model_cls.DMatrix(X_valid, feature_names=X.columns),
                                             ntree_limit=model.best_ntree_limit)
                if X_test is not None:
                    y_pred = model.predict(model_cls.DMatrix(X_test, feature_names=X.columns),
                                           ntree_limit=model.best_ntree_limit)

            else:
                train_data = model_cls.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
                model = model_cls.train(dtrain=train_data, num_boost_round=num_boost_round, params=params)
                y_pred_valid = model.predict(model_cls.DMatrix(X_valid, feature_names=X.columns))
                if X_test is not None:
                    y_pred = model.predict(model_cls.DMatrix(X_test, feature_names=X.columns))

        if model_type == 'cat':
            if early_stopping:
                test_size = early_stopping.get("test_size", .10)
                X_t_train, X_e_train, y_t_train, y_e_train = train_test_split(X_train, y_train, test_size = test_size)
                eval_metric = early_stopping.get("eval_metric", "MAE")
                use_best_model = early_stopping.get("use_best_model", True)
                model = model_cls(eval_metric=eval_metric, **params)
                model.fit(X_train, y_train, eval_set=(X_e_train, y_e_train), cat_features=[],
                          use_best_model=use_best_model, verbose=False)
            else:
                model = model_cls(**params)
                model.fit(X_train, y_train, cat_features=[], verbose=False)
            y_pred_valid = model.predict(X_valid)
            if X_test is not None:
                y_pred = model.predict(X_test)

        y_pred_valid = y_pred_valid.reshape(-1, )
        oof[valid_index] = y_pred_valid
        score = mean_absolute_error(y_valid, y_pred_valid)
        scores.append(score)

        print(f'Fold {fold_n}. MAE: {score:.4f}.')
        print('')

        if X_test is not None:
            y_pred = y_pred.reshape(-1, )
            prediction += y_pred

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    if X_test is not None:
        prediction /= n_fold

    print('CV mean score: {0:.4f}, mad: {1:.4f}.'.format(np.mean(scores), robust.mad(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');

    if X_test is not None:
        return np.mean(scores), robust.mad(scores), prediction
    else:
        return np.mean(scores), robust.mad(scores)
