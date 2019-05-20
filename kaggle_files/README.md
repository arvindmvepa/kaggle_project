# Saved Files

## Features

### 138 unscaled features for train/test set 
Generated from `Earthquakes FE. More features and samples` kaggle notebook. Train features are generated from 4194 independent contiguous segments of 150,000 time points. Train/test files are below:
- kaggle_project/kaggle_files/train/standard_138.csv
- kaggle_project/kaggle_files/test/standard_138_test.csv

### 138 standard normalized features for train/test set 
Generated from `Earthquakes FE. More features and samples` kaggle notebook. Train features are generated from 4194 independent contiguous segments of 150,000:
- kaggle_project/kaggle_files/train/standard_138_scaled.csv
- kaggle_project/kaggle_files/test/standard_138_scaled_test.csv

### Time to Failure (TTF) for 4194 independent contiguous train segments of 150,000 
Based on TTF for the last point in each 150,000 segment.
- kaggle_project/kaggle_files/train/ttf.csv


### Steven's ML4round1 for 4194 independent consecutive segments of 150k data points
- kaggle_project/kaggle_files/train/X_fillna_4195rows_996cols.csv
- kaggle_project/kaggle_files/test/Xtest_fillna_2624rows_996cols.csv

### Steven's ML4round4 for 4153 independent consecutive segments of 150k data points
split the train.csv into quake-based segments, each segment are chopped out at the "jump" of time_to_failure; 
the ML4_round4_quakes_explore.png describes how each segment looks like;
drop some NaN featrues; 
- kaggle_project/kaggle_files/train/X_quakebased_fillna_4153rows_984cols.csv
- kaggle_project/kaggle_files/test/Xtest_quakebased_fillna_2624rows_984cols.csv

### Steven's ML5round1 for 24000 independent consecutive segments of 150k data points
split the original train.csv into 6 segments, and sample 4000 out of each segment => total 24000 samples; 
use StandardScalar() to scale the Xtrain & Xvalid set, which produced the scaled_train_X.csv and scaled_test_X.csv in the folder

- kaggle_project/kaggle_files/train/ML5round1_scaled_train_X.csv
- kaggle_project/kaggle_files/test/ML5round1_scaled_test_X.csv

## Feature Selection

We applied the pearson correlation coefficient to each feature column for each feature set for train with the target vector for train. We removed features that had p-values below .05 and .01 respectively for each feature set.


## CV Results

The CV results are stored in `kaggle_files/cv_results`.

### exp1.csv

These results are generated from `notebooks/linear_mods_hyp_exp.ipynb`.

### exp2.csv

These results are generated from `notebooks/other_sup_mods_hyp_exp.ipynb`.

### exp3.csv

These results are generated from these experiments `experiments/lgb.yml`, `experiments/lgb_es.yml`, `experiments/xgb.yml`, `experiments/xgb_es.yml`, `experiments/cat.yml`, and `experiments/cat_es.yml` (there are fewer results from catboost due to long computation time).

### exp4_1.csv, exp4_2.csv, and exp4_3.csv

These are generated from `experiments/fs_test1.yml`, `experiments/fs_test2.yml`, and `experiments/fs_test3.yml`. These are mostly illustrative to show the updates to CV scoring and feature set options.