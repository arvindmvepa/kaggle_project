from exp.run import run_experiment_script

if __name__ == '__main__':
    # run_experiment_script(params="test.yml", num_searches=2, n_fold=2)
    # run_experiment_script(params="lgb.yml", num_searches=20, n_fold=10)
    # run_experiment_script(params="xgb.yml", num_searches=20, n_fold=10)
    run_experiment_script(params="cat.yml", num_searches=20, n_fold=10)
