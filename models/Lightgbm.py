import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc


class Lightgbm():
    def __init__(self, model_params=None):
        lgb_params = {
            'boosting_type': 'gbdt',
            'scale_pos_weight': 200,
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.2,
            'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
            'max_depth': 3,  # -1 means no limit
            'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
            'max_bin': 100,  # Number of bucketed bin for feature values
            'subsample': 0.7,  # Subsample ratio of the training instance.
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin': 200000,  # Number of samples for constructing bin
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha': 0,  # L1 regularization term on weights
            'reg_lambda': 0,  # L2 regularization term on weights
            'nthread': -1,
            'verbose': 0,
        }

        if model_params is None:
            self.model_params = lgb_params
        else:
            self.model_params = lgb_params
            self.model_params.update(model_params)

        self.model = None
        self.features = None
        self.best_iter = 0
        self.bst_score = 0

    # def fit(self, X_train, y_train, X_test, y_test, features, num_boost_round=3000, early_stopping_rounds=20):
    #    d_train = lgb.Dataset(X_train, label=y_train,feature_name=features)
    #    d_valid = lgb.Dataset(X_test, label=y_test,feature_name=features)
    #    self.model = lgb.train(self.model_params, d_train, num_boost_round=num_boost_round,\
    #                        valid_sets=[d_train, d_valid], valid_names=['train','valid'],\
    #                        early_stopping_rounds=early_stopping_rounds, evals_result={})

    def fit(self, X_train, y_train, ifcv=True):
        self.features = X_train.columns
        if ifcv:
            d_train = lgb.Dataset(X_train, label=y_train)
            self.model = lgb.train(self.model_params, d_train)
        else:
            x_train, x_valid, Y_train, Y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=66)
            d_train = lgb.Dataset(x_train, label=Y_train)
            del x_train, Y_train;
            gc.collect()
            d_valid = lgb.Dataset(x_valid, label=Y_valid)
            del x_valid, Y_valid;
            gc.collect()
            evals_results = {}
            self.model = lgb.train(self.model_params, d_train, num_boost_round=1000,
                                   valid_sets=[d_valid], valid_names=['valid'],
                                   early_stopping_rounds=25, evals_result=evals_results)
            self.best_iter = self.model.best_iteration
            self.bst_score = evals_results['valid']['auc'][self.best_iter-1]

    def predict(self, X, ifcv=True):
        if ifcv:
            return self.model.predict(X)
        else:
            return self.model.predict(X, num_iteration=self.best_iter)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    def get_features_importances(self):
        return pd.Series(data=self.model.feature_importance(), index=self.model.feature_name()).sort_values(
            ascending=False)

    def plot_features_importances(self):
        ax = lgb.plot_importance(self.model, max_num_features=100)
        plt.show()

    def return_best_score(self):
        return self.bst_score


best_lgb = {'boosting_type': 'gbdt',
            'scale_pos_weight': 500,
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.09784768292949085,
            'num_leaves': 13,
            'max_depth': 6,
            'min_child_samples': 41,
            'max_bin': 255,
            'subsample': 0.9,
            'subsample_freq': 1,
            'colsample_bytree': 0.6,
            'min_child_weight': 0,
            'subsample_for_bin': 200000,
            'min_split_gain': 0,
            'reg_alpha': 0.1147528557906301,
            'reg_lambda': 0.016517459948781423,
            'nthread': -1,
            'verbose': -1,
            'categorical_column': [0, 1, 2, 3, 4]}
