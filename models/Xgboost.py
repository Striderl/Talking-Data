import xgboost as xgb


class Xgboost():
    def __init__(self, model_params=None):
        xgb_params = {
            'eta': 0.6,
            'booster': 'gbtree',
            'tree_method': "hist",
            'grow_policy': "lossguide",
            'learning_rate': 0.1,
            'min_child_weight': 1,  # minimum sum of weights of all observations required in a child, controls overfitting
            'max_depth': 6,
            'subsample': 0.8,  # fraction of observations to be randomly samples for each tree.
            'colsample_bytree': 0.8,  # fraction of columns to be randomly samples for each tree. typical 0.5-1
            'scale_pos_weight': 200,  # data imbalance
            'gamma': 0,   # specifies the minimum loss reduction required to make a split,makes the algorithm conservative. typical 0-5
            'reg_lambda': 0,  # L2 regularization term on weights
            'alpha': 4,   # L1 regularization term on weight
            'objective': 'binary:logistic',  # No need for metric, it applies metric according to objective
            'eval_metric': 'auc',
            'nthread': -1
        }

        if model_params is None:
            self.model_params = xgb_params
        else:
            self.model_params = xgb_params
            self.model_params.update(model_params)

        self.model = None
        # self.model = xgb.XGBClassifier()
        # self.model.set_params(**self.model_params)

    def fit(self, X_train, y_train, **kwargs):
        dtrain = xgb.DMatrix(X_train.values, y_train.values)
        watchlist = [(dtrain, 'train')]
        self.model = xgb.train(self.model_params, dtrain, 100, watchlist, maximize=True, verbose_eval=20)
        # self.model.fit(X_train.values, y_train.values, **kwargs)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test.values)
        return self.model.predict(dtest)

    def get_params(self):
        return self.model_params if self.model_params is not None else {}

    @staticmethod
    def get_features_importances():
        return None

    @staticmethod
    def plot_features_importances():
        return None
