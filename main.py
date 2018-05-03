from feature import *
from tune import *
from models import Lightgbm, Catboost, Xgboost


# First step, decide what features to be included in the training process
feature_list = {
    "generated": [
        ("unique_channel_per_ip", unique_channel_per_ip, "int32"),
        ("unique_hour_per_ip_day", unique_hour_per_ip_day, "int32"),
        ("unique_app_per_ip", unique_app_per_ip, "int32"),
        ("unique_device_per_ip", unique_device_per_ip, "int32"),
        ("unique_channel_per_app", unique_channel_per_app, "int32"),
        ("unique_os_per_ip_app", unique_os_per_ip_app, "int32"),
        ("unique_app_per_ip_device_os", unique_app_per_ip_device_os, "int32"),
        ("seq_os_per_ip", seq_os_per_ip, "int32"),
        ("seq_app_per_ip_device_os", seq_app_per_ip_device_os, "int32"),
        ("count_channel_per_ip_day_hour", count_channel_per_ip_day_hour, "int32"),
        ("count_channel_per_ip_app", count_channel_per_ip_app, "int32"),
        ("count_channel_per_ip_app_os", count_channel_per_ip_app_os, "int32"),
        # ("nextclick", nextclick, "int32"),
        ("nextclick_online", nextclick_online, "float16"),
        # ("prevclick_online",prevclick_online,"float16"),
        ("var_hour_per_ip_day_channel", var_hour_per_ip_day_channel, "float16"),
        ("var_hour_per_ip_app_os", var_hour_per_ip_app_os, "float16"),
        ("var_day_per_ip_app_channel", var_day_per_ip_app_channel, "float16"),
        # ("mean_hour_per_ip_app_channel", mean_hour_per_ip_app_channel, "float16"),
        # ("seq_app_per_ip_device_os_channel", seq_app_per_ip_device_os_channel, "int32"),
        # ("count_app_per_ip_day_hour", count_app_per_ip_day_hour, "int32")
    ]
}

predictors = ['ip', 'app', 'device', 'os', 'channel']
for i in range(len(feature_list['generated'])):
    predictors.append(feature_list['generated'][i][0])

# get feature engineered train test sets
train_df, test_df = preprocess(144900000, 184900000, feature_list, pickle_folder="Pickle", debug=True)
# train_df, test_df = preprocess(0, 200000, feature_list, pickle_folder="Pickle/debug", debug=True)  # if debug mode

# if submit:
# train(train_df, test_df, Lightgbm.Lightgbm, predictors, record=True, submit=True, Model_params=Lightgbm.best_lgb, FOLDS=3, plot_feature_importance=True)

# if tune:
# lgb_trials = tune_single_model(train_df, test_df, Lightgbm.Lightgbm, predictors, space_lightgbm, max_evals=66, record=True)
catboost_trials = tune_single_model(train_df, test_df, Catboost.CatBoost, predictors, space_catboost, max_evals=1, record=True)
# xgb_trials = tune_single_model(train_df, test_df, Xgboost.Xgboost, predictors, space_xgb, max_evals=1, record=True)


