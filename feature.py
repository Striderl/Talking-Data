import pandas as pd
import numpy as np
import time
import os
import gc


def unique_channel_per_ip(df):
    return df[['ip']].merge(df[['ip', 'channel']].groupby(by='ip')['channel'].nunique().
                            reset_index().rename(columns={'channel': 'unique_channel_per_ip'}), how='left')['unique_channel_per_ip']


def unique_hour_per_ip_day(df):
    return df[['ip', 'day']].merge(df[['ip', 'day', 'hour']].groupby(by=['ip', 'day'])['hour'].nunique().
                                   reset_index().rename(columns={'hour': 'unique_hour_per_ip_day'}), how='left')['unique_hour_per_ip_day']


def unique_app_per_ip(df):
    return df[['ip']].merge(df[['ip', 'app']].groupby(by='ip')['app'].nunique().
                            reset_index().rename(columns={'app': 'unique_app_per_ip'}), how='left')['unique_app_per_ip']


def unique_device_per_ip(df):
    return df[['ip']].merge(df[['ip', 'device']].groupby(by='ip')['device'].nunique().
                            reset_index().rename(columns={'device': 'unique_device_per_ip'}), how='left')['unique_device_per_ip']


def unique_channel_per_app(df):
    return df[['app']].merge(df[['app', 'channel']].groupby(by='app')['channel'].nunique().
                             reset_index().rename(columns={'channel': 'unique_channel_per_app'}), how='left')['unique_channel_per_app']


def unique_os_per_ip_app(df):
    return df[['ip', 'app']].merge(df[['ip', 'app', 'os']].groupby(by=['ip', 'app'])['os'].nunique().
                                   reset_index().rename(columns={'os': 'unique_os_per_ip_app'}), how='left')['unique_os_per_ip_app']


def unique_app_per_ip_device_os(df):
    return df[['ip', 'device', 'os']].merge(
        df[['ip', 'device', 'os', 'app']].groupby(by=['ip', 'device', 'os'])['app'].nunique().
        reset_index().rename(columns={'app': 'unique_app_per_ip_device_os'}), how='left')['unique_app_per_ip_device_os']


def seq_os_per_ip(df):
    return df[['ip', 'os']].groupby(by='ip')['os'].cumcount()


def seq_app_per_ip_device_os(df):
    return df[['ip', 'device', 'os', 'app']].groupby(by=['ip', 'device', 'os'])['app'].cumcount()


def count_channel_per_ip_day_hour(df):
    return df[['ip', 'day', 'hour']].merge(
        df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])['channel'].count().
        reset_index().rename(columns={'channel': 'count_channel_per_ip_day_hour'}), how='left')['count_channel_per_ip_day_hour']


def count_channel_per_ip_app(df):
    return df[['ip', 'app']].merge(df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])['channel'].count().
                                   reset_index().rename(columns={'channel': 'count_channel_per_ip_app'}), how='left')['count_channel_per_ip_app']


def count_channel_per_ip_app_os(df):
    return \
    df[['ip', 'app', 'os']].merge(df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])['channel'].count().
                                  reset_index().rename(columns={'channel': 'count_channel_per_ip_app_os'}), how='left')['count_channel_per_ip_app_os']


# def nextclick(df):
#     time_diff = df[['ip','app','device','os']].merge(df[['ip','app','device','os','click_time']].groupby(by=['ip','app','device','os']).nth(1).reset_index().\
#             rename(columns={'click_time':'next_click'}),how = 'left')['next_click']-df['click_time']
#     cond = np.isnat(time_diff)
#     return pd.Series([99999 if cond[i] else timedelta.total_seconds(time_diff[i]) for i in range(len(time_diff))]).astype(np.int32)


# from https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
def nextclick_online(df):
    D = 2 ** 26
    cat = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str)
           + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)

    epochtime = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, t in zip(reversed(cat.values), reversed(epochtime.values)):
        next_clicks.append(click_buffer[category] - t)
        click_buffer[category] = t
    return pd.Series(list(reversed(next_clicks)))


def prevclick_online(df):
    D = 2 ** 26
    cat = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str)
           + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)

    epochtime = df['click_time'].astype(np.int64) // 10 ** 9
    prev_clicks = []
    for category, t in zip(cat.values, epochtime.values):
        prev_clicks.append(click_buffer[category] - t)
        click_buffer[category] = t
    return pd.Series(list(prev_clicks))


def var_hour_per_ip_day_channel(df):
    return df[['ip', 'day', 'channel']].merge(
        df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'channel'])['hour'].var().
        reset_index().rename(columns={'hour': 'var_hour_per_ip_day_channel'}), how='left')['var_hour_per_ip_day_channel']


def var_hour_per_ip_app_os(df):
    return df[['ip', 'app', 'os']].merge(df[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])['hour'].var().
                                         reset_index().rename(columns={'hour': 'var_hour_per_ip_app_os'}), how='left')['var_hour_per_ip_app_os']


def var_day_per_ip_app_channel(df):
    return df[['ip', 'app', 'channel']].merge(
        df[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])['day'].var(). \
        reset_index().rename(columns={'day': 'var_day_per_ip_app_channel'}), how='left')['var_day_per_ip_app_channel']


def preprocess(begin, end, feature_list, pickle_folder="Pickle", debug=False):
    dtypes = {'ip': 'category',  # 'uint32',
              'app': 'category',
              'device': 'category',
              'os': 'category',
              'channel': 'category',
              'is_attributed': 'uint8',
              'click_id': 'uint32'
             }
    print('loading train data from row '+str(begin)+' to row '+str(end))
    train_df = pd.read_csv("data/train.csv", parse_dates=['click_time'], skiprows=range(1, begin),
                           nrows=end - begin, dtype=dtypes, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
    print('loading test data........................')
    if not debug:
        test_df = pd.read_csv("data/test.csv", parse_dates=['click_time'], dtype=dtypes,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("data/test.csv", parse_dates=['click_time'], dtype=dtypes, nrows=1000,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

    print('start feature engineering and loading features already exists.........')
    start_time = time.time()
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    test_df['hour'] = pd.to_datetime(test_df.click_time).dt.hour.astype('uint8')
    test_df['day'] = pd.to_datetime(test_df.click_time).dt.day.astype('uint8')

    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)
    for feature_name, feature_fcn, feature_dtype in feature_list['generated']:
        # for train features:
        train_pickle_name = 'train_' + feature_name + '_pickle' + '%d_%d' % (begin, end)
        train_pickle_path = pickle_folder + "/" + train_pickle_name
        if os.path.exists(train_pickle_path):
            print("loading pickle "+train_pickle_path)
            feature = pd.read_pickle(train_pickle_path)
        else:
            feature = feature_fcn(train_df)
            print("creating pickle "+train_pickle_path)
            feature.to_pickle(train_pickle_path)
        train_df[feature_name] = feature
        train_df[feature_name] = train_df[feature_name].astype(feature_dtype)
        del feature; gc.collect()

        # for test features:
        test_pickle_name = 'test_' + feature_name +'_pickle'
        test_pickle_path = pickle_folder + "/" + test_pickle_name
        if os.path.exists(test_pickle_path):
            print("loading pickle "+test_pickle_path)
            feature = pd.read_pickle(test_pickle_path)
        else:
            feature = feature_fcn(test_df)
            print("creating pickle "+test_pickle_path)
            feature.to_pickle(test_pickle_path)
        test_df[feature_name] = feature
        test_df[feature_name] = test_df[feature_name].astype(feature_dtype)
        del feature; gc.collect()

    print("finished preprocessing within "+str(time.time()-start_time)+"s")
    return train_df, test_df


#      click_time
# day
# 6       9308568   0 - 9308568
# 7      59633310   9308568 - 68941878
# 8      62945075   68941878 - 131886953
# 9      53016937   131886953 - 184903890

