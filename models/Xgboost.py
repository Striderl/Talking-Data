import xgboost as xgb
from sklearn.model_selection import train_test_split
import gc


class Xgboost():
    def __init__(self, model_params=None):
        xgb_params = {
            'eta': 0.6,
            'booster': 'gbtree',
            # 'tree_method': "hist",
            # 'grow_policy': "lossguide",
            'learning_rate': 0.1,
            'min_child_weight': 1,  # minimum sum of weights of all observations required in a child, controls overfitting
            'max_depth': 5,
            'subsample': 0.8,  # fraction of observations to be randomly samples for each tree.
            'colsample_bytree': 0.8,  # fraction of columns to be randomly samples for each tree. typical 0.5-1
            'scale_pos_weight': 200,  # data imbalance
            'gamma': 0,   # specifies the minimum loss reduction required to make a split,makes the algorithm conservative. typical 0-5
            'reg_lambda': 0,  # L2 regularization term on weights
            'alpha': 4,   # L1 regularization term on weight
            'objective': 'binary:logistic',  # No need for metric, it applies metric according to objective
            'eval_metric': 'auc',
            'silent': 1,
            'nthread': -1
        }

        if model_params is None:
            self.model_params = xgb_params
        else:
            self.model_params = xgb_params
            self.model_params.update(model_params)

        self.model = None

    def fit(self, X_train, y_train, ifcv=True):
        if ifcv:
            dtrain = xgb.DMatrix(X_train.values, y_train.values)
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(self.model_params, dtrain, 200, watchlist, maximize=True, verbose_eval=5)
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=66)
            dtrain = xgb.DMatrix(x_train.values, y_train.values)
            del x_train, y_train; gc.collect()
            dvalid = xgb.DMatrix(x_valid.values, y_valid.values)
            del x_valid, y_valid; gc.collect()
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            self.model = xgb.train(self.model_params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=20)
            return self.model

    def predict(self, x_test, ifcv=True):
        dtest = xgb.DMatrix(x_test.values)
        if ifcv:
            return self.model.predict(dtest)
        else:
            return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    def get_features_importances(self):
        return None

    def plot_features_importances(self):
        self.model.plot_importance()


best_xgb = {
            'eta': 0.6,
            'booster': 'gbtree',
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'learning_rate': 0.26069454841714973,
            'min_child_weight': 1,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 200,
            'gamma': 2,
            'reg_lambda': 0.6323212359081647,
            'alpha': 4,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': -1,
            'reg_alpha': 0.1744641464642278
}
