from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split
from sklearn.svm._libsvm import cross_validation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils
from tensorflow.examples.saved_model.integration_tests.mnist_util import INPUT_SHAPE
from statsmodels.tsa import stattools
from tensorflow.python.keras.models import Model
from itertools import zip_longest

from NTI_srv_neyronka import models as neyron_models
import pandas as pd
import numpy as np

import datetime
from itertools import zip_longest
import matplotlib.pyplot as plt


test_set_size = 0.1


def train_test_valid_split_plus_scaling(df, valid_set_size, test_set_size):

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    df_copy = df.reset_index(drop=True)

    df_test = df_copy.iloc[int(np.floor(len(df_copy) * (1 - test_set_size))):]
    df_train_plus_valid = df_copy.iloc[: int(np.floor(len(df_copy) * (1 - test_set_size)))]

    df_train = df_train_plus_valid.iloc[: int(np.floor(len(df_train_plus_valid) * (1 - valid_set_size)))]
    df_valid = df_train_plus_valid.iloc[int(np.floor(len(df_train_plus_valid) * (1 - valid_set_size))):]

    X_train = df_train.iloc[:, 0:]
    X_valid = df_valid.iloc[:, 0:]
    X_test = df_test.iloc[:, 0:]

    y_train = X_train.pop('WS50M')
    y_valid = X_valid.pop('WS50M')
    y_test = X_test.pop('WS50M')



    # X_train.pop('PS')
    # X_valid.pop('PS')
    # X_test.pop('PS')

    # global Target_scaler

    # Target_scaler = MinMaxScaler(feature_range=(0, 0.2))
    # Feature_scaler = MinMaxScaler(feature_range=(0, 0.2))
    #
    # X_train_scaled = Feature_scaler.fit_transform(np.array(X_train))
    # X_valid_scaled = Feature_scaler.fit_transform(np.array(X_valid))
    # X_test_scaled = Feature_scaler.fit_transform(np.array(X_test))
    #
    # y_train_scaled = Target_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    # y_valid_scaled = Target_scaler.fit_transform(np.array(y_valid).reshape(-1, 1))
    # y_test_scaled = Target_scaler.fit_transform(np.array(y_test).reshape(-1, 1))

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def run_model_on_test_set(model, df, X_test, y_test, max_y, min_y):
    # Recall that we have already defined "Target_scaler" as global in the 'split n scale' function earlier

    y_pred = model.predict(X_test)
    for i in range(0, y_pred.size):
        y_pred[i] = y_pred[i] * max_y+min_y
    y_test = y_test*max_y+min_y

    # y_pred_rescaled = Target_scaler.inverse_transform(y_pred)
    #
    # y_test_rescaled = Target_scaler.inverse_transform(y_test)

    from sklearn.metrics import r2_score
    # score = r2_score(y_test_rescaled, y_pred_rescaled)

    y_actual = pd.DataFrame(y_test, columns=['Actual'])
    y_test = y_test.sort_index().reset_index(drop=True)
    # y_actual.set_index(df.index[int(np.floor((1 - test_set_size) * len(df))):], inplace=True)

    y_hat = pd.DataFrame(y_pred, columns=['Predicted'])
    # y_hat.set_index(df.index[int(np.floor((1 - test_set_size) * len(df))):], inplace=True)

    # y_hat = stattools.acf(y_hat, unbiased=True, nlags=1000)
    # y_actual = stattools.acf(y_actual, unbiased=True, nlags=1000)


    # plt.figure(figsize=(7, 5))
    plt.subplot(2, 1, 1)
    plt.plot(y_test, linestyle='solid', color='r')  # plotting only a few values for better visibility
    plt.subplot(2, 1, 2)
    plt.plot(y_hat, linestyle='dashed', color='b')
    plt.show()


class Command(BaseCommand):
    def handle(self, *args, **options):
        dataframe = pd.read_csv('NTI_srv_neyronka/management/commands/POWER_SP.csv')
        years = dataframe['YEAR']
        months = dataframe.pop('MO')
        days = dataframe.pop('DY')
        dates = []
        data_plt = []
        for year,  month, day in zip_longest(years, months, days):
            dates.append('%s-%s-%s 00:00' % (year, month, day))
            data_plt.append(datetime.datetime(year, month, day))
        dates = pd.DataFrame({'date': dates})

        dataframe['YEAR'] = dates
        dataframe.rename(columns={'YEAR': 'DATE'},  inplace=True)
        ALLSKY_SFC_SW_DWN = dataframe['ALLSKY_SFC_SW_DWN']
        remove_row = []
        for i in range(len(ALLSKY_SFC_SW_DWN)):
            if ALLSKY_SFC_SW_DWN[i] < 0:
                remove_row.append(i)

        T2M = dataframe['T2M']
        for i in range(len(T2M)):
            if T2M[i] < -6 and i not in remove_row :
                remove_row.append(i)

        dataframe.drop(index=remove_row, inplace=True)
        dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
        dataframe.set_index(['DATE'], drop=True, inplace=True)
        dataframe.drop(columns=['LAT', 'LON'], inplace=True)

        x_train, x_valid, x_test, y_train, y_valid, y_test = train_test_valid_split_plus_scaling(dataframe,
                                                                                                 0.2,
                                                                                                 test_set_size)

        mean = x_train.mean(axis=0)
        x_train -= mean

        std = x_train.std(axis=0)
        x_train/=std

        def norm(x):
            return (x-mean)/ std

        x_train = norm(x_train)
        x_test = norm(x_test)

        min_y = y_train.min()
        max_y = y_train.max() - min_y

        y_train = (y_train-min_y)/max_y
        y_test = (y_test-min_y)/max_y

        input_layer = Input(shape=5)
        dense1 = Dense(8, activation='relu')(input_layer)
        dense2 = Dense(32, activation='relu')(dense1)
        # dropout_layer = Dropout(0.4)(dense2)
        # dense3 = Dense(10, activation='sigmoid')(dropout_layer)
        # dense4 = Dense(5, activation='relu')(dense3)
        output_layer = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mse',
                      optimizer='rmsprop',
                      metrics=['Accuracy', 'Recall'])
        model.summary()
        # neyron_models.Settings(name='actual', std=list(std), mean=list(mean), max_y=max_y, min_y=min_y).save()

        model.fit(x_train, y_train,
                            batch_size=2, epochs=80,
                            verbose=True, validation_data=(x_valid, y_valid))

        run_model_on_test_set(model, dataframe, x_test, y_test, max_y,min_y)
        model.save('model12.h5')