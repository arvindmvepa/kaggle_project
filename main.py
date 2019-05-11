from exp.run import run_experiment_script

if __name__ == '__main__':
    # run_experiment_script(params="test.yml", num_searches=2, n_fold=2)
    # run_experiment_script(params="lgb.yml", save_results="exp3_1.csv", num_searches=40, n_fold=10)
    # run_experiment_script(params="xgb.yml", save_results="exp3_1.csv", num_searches=40, n_fold=10)
    # run_experiment_script(params="cat.yml", save_results="exp3_1.csv", num_searches=40, n_fold=10)
    run_experiment_script(params="lgb_es.yml", save_results="exp3_1.csv", num_searches=40, n_fold=10)
    run_experiment_script(params="xgb.yml", save_results="exp3_1.csv", num_searches=40, n_fold=10)
    run_experiment_script(params="cat.yml", save_results="exp3_1.csv", num_searches=40, n_fold=10)
