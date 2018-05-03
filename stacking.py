from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials
import datetime
import os
import pandas as pd
import gc
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time


def get_stacking_features(validation=True):
    if validation:
        pickle_path = "Pickle/stacking/validation"
    else:
        pickle_path = "Pickle/stacking/test"
    all_pickles = os.listdir(pickle_path)

    stacking_list = [pd.read_csv(os.path.join(pickle_path, f), index_col=0) for f in all_pickles]
    stacking_df = pd.concat(stacking_list, axis=1)
    cols = list(map(lambda x: "is_attributed_" + str(x), range(len(stacking_df.columns))))
    stacking_df.columns = cols
    stacking_df.reset_index(inplace=True)
    return stacking_df


def generate_stacking_features(train_df, test_df, predictors, Model, params=None, target='is_attributed'):
    stacking_pickle_path = 'Pickle/stacking'
    if not os.path.exists(stacking_pickle_path):
        os.makedirs(stacking_pickle_path)
        os.makedirs(stacking_pickle_path + '/validation')
        os.makedirs(stacking_pickle_path + '/test')

    time = datetime.datetime.now().strftime("%H-%M-%S")
    print('fitting model: ' + str(Model.__name__))
    model = Model(model_params=params)
    model.fit(train_df[predictors], train_df[target])

    train_pickle_name = stacking_pickle_path + '/validation/' + str(Model.__name__) + '_train_stacking_' + str(time)
    print('creating pickle: ' + train_pickle_name)
    stack_train = pd.DataFrame()
    stack_train['ip'] = train_df['ip'].astype('uint32')
    stack_train[target] = model.predict(train_df[predictors])
    stack_train.to_csv(train_pickle_name+'.csv', index=False)
    del stack_train; gc.collect()

    test_pickle_name = stacking_pickle_path + '/test/' + str(Model.__name__) + '_test_stacking_' + str(time)
    print('creating pickle: ' + test_pickle_name)
    stack_test = pd.DataFrame()
    stack_test['click_id'] = test_df['click_id'].astype('int')
    stack_test[target] = model.predict(test_df[predictors])
    stack_test.to_csv(test_pickle_name+'.csv', index=False)
    print('done with generating %s stacking pickles...'%Model.__name__)
    del stack_test; gc.collect()


def train_stacking(stacking_df, Model, params=None, FOLDS=3, record=False, submit=False, **kwargs):

    predictors = stacking_df.columns[1:]
    train_target = pd.read_csv('Pickle/stacking/validation/validation_target.csv')  # need revision

    if not submit:
        if not os.path.exists('Error'):
                os.makedirs('Error')
        time = datetime.datetime.now().strftime("%H-%M-%S")
        if record:
            train_recorder = open('Error/stacking_%s_%s_params.txt' %(Model.__name__, time), 'w')
            train_recorder.write(Model.__name__ + '\n')

        kfold_errors = []
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=66)
        print("start %d folds train....." %(FOLDS))
        for i, (train_index, valid_index) in enumerate(kf.split(stacking_df)):
            X_train = stacking_df.iloc[train_index][predictors]
            y_train = train_target.iloc[train_index]['is_attributed']

            X_valid = stacking_df.iloc[valid_index][predictors]
            y_valid = train_target.iloc[valid_index]['is_attributed']

            print(X_train.shape, y_train.shape)
            print('training for fold %d.......'%(i))
            model = Model(model_params=params)
            model.fit(X_train, y_train, **kwargs)
            train_auc = roc_auc_score(y_train, model.predict(X_train))
            print("fold train auc: ", str(train_auc))

            print('validating for fold %d......'%(i))
            valid_auc = roc_auc_score(y_valid, model.predict(X_valid))
            kfold_errors.append(valid_auc)
            print("fold validation auc: ", str(valid_auc))

            if record:
                train_recorder.write('\nFold %d\n' %i)
                train_recorder.write('Parameters: %s\n' % model.get_params())
                feature_importances = model.get_features_importances
                if feature_importances is not None:
                    train_recorder.write('Feature importances:\n%s\n' %feature_importances)
                train_recorder.write('Train error: ' + str(train_auc) + '\n')
                train_recorder.write('Validation error: ' + str(valid_auc) + '\n')

            print("------------------------------------")

        avg_fold_auc = np.mean(kfold_errors)

        if record:
            train_recorder.write("\nAverage cross validation mean auc: %f\n" %avg_fold_auc)
            train_recorder.close()
        print("average cross validation mean auc %f\n\n" % avg_fold_auc)

        with open('Error/experiments.txt', 'a') as record:
            record.write('\n\nTime: %s\n' %time)
            record.write('model_params:%s\n' %model.get_params())
            record.write('local validtion mean auc: %f\n' % avg_fold_auc)

        return avg_fold_auc

    else:  # if submit
        print("getting test stacking features....")
        test_target = pd.read_csv('Pickle/stacking/test/test_target.csv')  # need revision
        test_df = get_stacking_features(validation=False)
        if not os.path.exists("Submission"):
            os.makedirs("Submission")

        model = Model(model_params=params)
        model.fit(stacking_df[predictors], train_target['is_attributed'], **kwargs)
        sub = pd.DataFrame()
        sub['click_id'] = test_target['click_id'].astype('int')
        sub['is_attributed'] = model.predict(test_df[predictors])
        time = datetime.datetime.now().strftime("%H-%M-%S")
        sub.to_csv('Submission/stacking_sub_it_%s.csv'%(time), index=False, float_format='%.9f')


def tune_stacking(stacking_df, meta_model, parameter_space, max_evals=100, trials=None, **kwargs):

    def tune_wrapper(params):
        print(params)
        loss = - train_stacking(stacking_df, meta_model, params=params, record=True, submit=False, **kwargs)
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
    best = fmin(tune_wrapper, parameter_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    t2 = time.time()
    print('\nbest trial get at round: ' + str(trials.best_trial['tid']))
    print('best loss: ' + str(trials.best_trial['result']['loss']))
    print(best)
    print(space_eval(parameter_space, best))
    print("time: %s s" %((t2-t1)))
    return trials
