import os
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask import dataframe as dd
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np
import optuna
import gc
import logging

num_round = 1000

def objective(client, dtrain, dtest, test_y, trial):
        
    params = {
        'objective': trial.suggest_categorical('objective',['reg:squarederror']), 
        'tree_method': trial.suggest_categorical('tree_method',['gpu_hist']),  # 'gpu_hist','hist'
        'lambda': trial.suggest_loguniform('lambda',1e-3,10.0),
        'alpha': trial.suggest_loguniform('alpha',1e-3,10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3,1.0),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001,0.1),
        #'n_estimators': trial.suggest_categorical('n_estimators', [1000]),
        'max_depth': trial.suggest_categorical('max_depth', [3,5,7,9,11,13,15,17,20]),
        #'random_state': trial.suggest_categorical('random_state', [24,48,2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1,300),
        'eval_metric': trial.suggest_categorical('eval_metric',['rmse']),

    }

    output = xgb.dask.train(client, params, dtrain, num_round)
    
    booster = output['booster']  # booster is the trained model
    booster.set_param({'predictor': 'gpu_predictor'})

    predictions = xgb.dask.predict(client, booster, dtest)
    
    predictions = predictions.compute()
    
    predictions = np.clip(predictions, 0.12, None)

    rmsle = mean_squared_log_error(test_y, predictions, squared=False)
    
    return rmsle

def main():
    train_x = dd.read_csv('../input/xgtrain_0.csv')
    test_x = dd.read_csv('../input/xgval_0.csv')

    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    train_y = train_x['target'] 
    train_x = train_x[train_x.columns.difference(['target'])]

    test_x = test_x.replace([np.inf, -np.inf], np.nan)
    test_y = test_x['target']
    test_x = test_x[test_x.columns.difference(['target'])]

    with LocalCUDACluster(CUDA_VISIBLE_DEVICES=["GPU-a19c00c3-2832-fe38-1c43-c18db3e909da", "GPU-58b97c92-e879-49d3-85b5-1d9615f10873", "GPU-d21cfed4-2e1a-f313-839c-ea008aca027a", "GPU-e3b349d7-ac6c-77ab-3564-ed9d05d50bac"]) as cluster:
        client = Client(cluster)
        dtrain = xgb.dask.DaskDMatrix(client, train_x, train_y)
        dtest = xgb.dask.DaskDMatrix(client, test_x, test_y)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler("optuna_xgb_output_1000_0.log", mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

        study = optuna.create_study(direction='minimize', storage="sqlite:///xgb_optuna_tests.db", study_name="jul_2021_test_1000_0")
        logger.info("Start optimization.")
        study.optimize(lambda trial: objective(client, dtrain, dtest, test_y, trial), n_trials=1000)
        
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df.to_csv('optuna_xgb_output_1000_0.csv', index=False)

if __name__ == "__main__":
    main()



