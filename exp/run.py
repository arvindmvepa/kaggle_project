import pandas as pd
import json
import os
import yaml
import warnings
from exp.train import train_model
from exp.hyp.search import random_search, grid_search
from exp.features import load_train_features
warnings.filterwarnings("ignore")


def run_experiment(X, Y, alg, alg_params, n_fold=5, X_test=None, score_df=None, save_results=None, search_type="random",
                   num_searches=100):
    """
    This runs a hyper-parameter search experiment.
    Parameters
    ----------
    alg : str
        A string which represents the algorithm to do hyper-parameter search for.
    alg_params: dict
        A dictionary, where the key is the hyper-parameter, and the value is a list of possible hyper-parameter values.
    n_fold: int
        The number of folds used for cross-validation.
    score_df : Pandas.DataFrame or None
        If None, an empty DataFrame is created with "alg", "score", "mad", and "params_json" columns. Otherwise, a
        DataFrame of scores from previous experiment(s) is added to during the experiment
    save_results : str
        If not None, the file to save experiment results to
    search_type : str
        Choices are `random` or `grid`, representing random search and grid search respectively.
    num_searches : int
        The number of hyper-parameter searches (only applicable for random search)
    Returns
    -------
    Pandas.DataFrame
        A DataFrame of scores from the current experiment potentially with scores from previous experiment(s).
    """
    if score_df is None:
        score_df = pd.DataFrame({}, columns=["alg", "score", "mad", "params_json"])
    elif isinstance(score_df, str):
        score_df = pd.read_csv(score_df)
    if isinstance(save_results, str):
        if not os.path.exists(save_results):
            # save `score_df` (including headers) to file if file doesn't exist
            score_df.to_csv(save_results, index=False)
    # different search options to generate hyper-parameter experiments
    if search_type == "random":
        param_searches = random_search(num_searches=num_searches, **alg_params)
    if search_type == "grid":
        param_searches = grid_search(**alg_params)

    # run experiment
    for params in param_searches:
        # instantiate model from hyper-parameters
        # model = alg_cls(**param_search)
        # debug
        print(params)
        # produce cv score and mad
        score, mad = train_model(X=X, Y=Y, X_test=X_test, n_fold=n_fold, params=params, alg=alg)
        # generate dataframe row to track alg scores
        df_ = pd.DataFrame(
            {"alg": [alg], "score": [score], "mad": [mad], "params_json": [json.dumps(params, sort_keys=True)]})
        # save to csv file after each search completes
        if isinstance(save_results, str):
            df_.to_csv(save_results, index=False, header=False, mode="a")
        # append to overall dataframe
        score_df = score_df.append(df_)
    return score_df


def run_experiment_script(params, search_type="random", num_searches=20, n_fold=10, save_results="exp.csv"):
    """
    This is a script for running an experiment, also including creating the features and iterating through algs' params.
    This primarily allows you to run this on script without having to worry about using a jupyter notebook.
    Parameters
    ----------
    params : str or dict
        If str, then a yml file which represents the ensuing dictionary description. If it is a dictionary the keys are
        keys in exp.mappings.alg_map and the values are dictionaries, where the key is the hyper-parameter, and the
        value is a list of possible hyper-parameter values.
    search_type : str
        Choices are `random` or `grid`, representing random search and grid search respectively.
    num_searches : int
        The number of hyper-parameter searches (only applicable for random search)
    n_fold: int
        The number of folds used for cross-validation.
    save_results : str
        If not None, the file to save experiment results to
    """

    X, y_tr = load_train_features("standard_scaled")

    # load params yaml file
    if isinstance(params, str):
        with open(params, 'r') as stream:
            params = yaml.load(stream)

    for alg in params.keys():
        print(alg)
        run_experiment(X=X, Y=y_tr, n_fold=n_fold, alg=alg, alg_params=params[alg],  search_type=search_type,
                       num_searches=num_searches, save_results=save_results)
