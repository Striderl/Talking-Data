import time
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials
from train import *
import datetime


space_lightgbm = {
    'learning_rate': hp.loguniform('learning_rate', -3, -1),
    'scale_pos_weight': hp.choice('scale_pos_weight', [200, 500]),
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': hp.choice('num_leaves', list(range(5, 15, 2))),
    'max_depth': hp.choice('max_depth', list(range(2, 7, 1))),
    'min_child_samples': hp.choice('min_child_samples', list(range(1, 101, 20))),
    'max_bin': hp.choice('max_bin', list(range(50, 151, 10))),
    'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9]),
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.7, 0.8, 0.9]),
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'reg_alpha': hp.loguniform('reg_alpha', -6, 0),
    'reg_lambda': hp.loguniform('reg_lambda', -6, 0),
    'verbose': -1,
}

space_rf = {
    'n_estimators': hp.choice('n_estimators', list(range(100, 1000, 100))),
    'criterion': hp.choice('criterion', ['mae', 'mse']),
    'max_features': hp.loguniform('max_features', -2, -1),
    'max_depth': hp.choice('max_depth', list(range(3, 8))),
    'min_samples_split': hp.loguniform('min_samples_split', -4, -2),
    'min_samples_leaf': hp.loguniform('min_samples_leaf', -4, -2),
    'n_jobs': -1
}

space_catboost = {
    'iterations': 200,
    'learning_rate': hp.loguniform('learning_rate', -4, -1),
    'depth': hp.choice('depth', list(range(3, 7, 1))),
    'l2-leaf-reg': hp.loguniform('l2-leaf-reg', 0, 2),
    'scale_pos_weight': 200,
    # 'border_count': hp.choice('border_count', [16]),
    # 'feature_border_type': 'MinEntropy',
    'leaf_estimation_iterations': hp.choice('leaf_estimation_iterations', [5, 10, 20]),
    'leaf_estimation_method': 'Gradient',
    'boosting_type': 'Plain',
}

space_xgb = {
    'booster': 'gbtree',
    'learning_rate': hp.loguniform('learning_rate', -3, -1),
    'min_child_weight': hp.choice('min_child_weight', list(range(3))),  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'max_depth': hp.choice('max_depth', list(range(3, 7, 1))),
    'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.7, 0.8, 0.9]),
    'scale_pos_weight': 200,  # data imbalance
    'gamma': hp.choice('gamma', list(range(4))),
    'reg_lambda': hp.loguniform('reg_lambda', -6, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -6, 0),
    'objective': 'binary:logistic',
    'nthread': -1
}


def tune_single_model(train_df, test_df, Model, predictors, parameter_space, max_evals=100, trials=None, folds=3, **kwargs):

    def train_wrapper(params):
        print(params)
        loss = - train(train_df, test_df, Model, predictors, Model_params=params, FOLDS=folds, **kwargs)
        return {
            'loss': loss,
            'status': STATUS_OK,
            'eval_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': params
        }

    if trials is None:
        trials = Trials()
    # tuning parameters
    t1 = time.time()
    best = fmin(train_wrapper, parameter_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    t2 = time.time()
    print('\nbest trial get at round: ' + str(trials.best_trial['tid']))
    print('best loss: ' + str(trials.best_trial['result']['loss']))
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s s" %((t2-t1)))
    return trials



