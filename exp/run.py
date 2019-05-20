import pandas as pd
import json
import os
import yaml
import warnings
from exp.train import train_model
from exp.hyp.search import random_search, grid_search
import copy
warnings.filterwarnings("ignore")


def run_experiment(X, Y, alg, alg_params, n_fold=10, shuffle=True, rs=None, score_df=None, save_results=None,
                   search_type="random", num_searches=100):
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
    shuffle: bool
        Whether to shuffle the data when doing cross-validation
    rs: None or int
        If not `None`, random seed used for shuffling for cross-validation.
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
    cv_score_string = "cv_score_shuffle_" + str(shuffle) + "_rs_" + str(rs)
    null_df = pd.DataFrame({}, columns=["alg", "feature_set", cv_score_string, "mad", "params_json"])
    if score_df is None:
        score_df = null_df
    elif isinstance(score_df, str):
        loaded_score_df = pd.read_csv(score_df)
        loaded_columns = set(loaded_score_df.columns)
        current_columns = set(null_df.columns)
        if loaded_columns != current_columns:
            raise ValueError("Loaded score_df has unexpected columns")
        else:
            score_df = loaded_score_df
    if isinstance(save_results, str):
        if not os.path.exists(save_results):
            # save `score_df` (including headers) to file if file doesn't exist
            score_df.to_csv(save_results, index=False)
        else:
            load_results = pd.read_csv(save_results)
            loaded_columns = set(load_results.columns)
            current_columns = set(score_df.columns)
            if loaded_columns != current_columns:
                raise ValueError("save_results has unexpected columns")

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
        params = copy.deepcopy(params)
        # extract feature set used
        fs = params.pop("fs", "standard_scaled")
        # produce cv score and mad
        score, mad = train_model(fs=fs, n_fold=n_fold, shuffle=shuffle, rs=rs, params=params, alg=alg)
        # generate dataframe row to track alg scores
        df_ = pd.DataFrame(
            {"alg": [alg], "feature_set": fs, cv_score_string: [score], "mad": [mad],
             "params_json": [json.dumps(params, sort_keys=True)]})
        # save to csv file after each search completes
        if isinstance(save_results, str):
            df_.to_csv(save_results, index=False, header=False, mode="a")
        # append to overall dataframe
        score_df = score_df.append(df_)
    return score_df


def run_experiment_script(params, search_type="random", num_searches=20, n_fold=10, shuffle=True, rs=None,
                          save_results="exp.csv"):
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

    # load params yaml file
    if isinstance(params, str):
        with open(params, 'r') as stream:
            params = yaml.load(stream)

    for alg in params.keys():
        print(alg)
        run_experiment(n_fold=n_fold, shuffle=shuffle, rs=rs, alg=alg, alg_params=params[alg],
                       search_type=search_type, num_searches=num_searches, save_results=save_results)
