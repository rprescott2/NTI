from sklearn.preprocessing import MinMaxScaler

from NTI_srv_neyronka import models as neyron_model
from tensorflow.python.keras.models import Model, load_model
import pandas as pd
import numpy as np

settings = neyron_model.Settings.objects.get(name='actual')
mean = np.array(settings.mean.replace('[', '').replace(']', '').split(',')[0:-1]).astype(float).reshape(1,-1)
std = np.array(settings.std.replace('[', '').replace(']', '').split(',')[0:-1]).astype(float).reshape(1,-1)
min_y = float(settings.min_y)
max_y = float(settings.max_y)


def norm_wind(x, mean, std):
    return (x - mean) / std


def pack_wind(data):
    global mean,std

    data -= mean
    data /= std
    data = norm_wind(data, mean, std)
    return data


def pack_allsky(data,Feature_scaler):
    scaled_data = Feature_scaler.transform(np.array(data))
    return scaled_data


def allsky_predict(data, Feature_scaler):
    model = load_model('model1.h5')
    result = model.predict(data)
    result = Feature_scaler.inverse_transform(result)
    return result


def wind_predict(data):
    model = load_model('model11.h5')
    result = model.predict(data)
    for i in range(0, result.size):
        result[i] = result[i] * max_y+min_y
    return result


def fit_scalers(data):

    df_copy = data.reset_index(drop=True)
    df_test = df_copy.iloc[int(np.floor(len(df_copy) * (1 - 0.1))):]
    df_train_plus_valid = df_copy.iloc[: int(np.floor(len(df_copy) * (1 - 0.1)))]

    df_train = df_train_plus_valid.iloc[: int(np.floor(len(df_train_plus_valid) * (1 - 0.2)))]
    df_valid = df_train_plus_valid.iloc[int(np.floor(len(df_train_plus_valid) * (1 - 0.2))):]

    data = df_train.iloc[:, 0:]

    target = data.pop('ALLSKY_SFC_SW_DWN')

    Target_scaler = MinMaxScaler(feature_range=(0, 0.1))
    Feature_scaler = MinMaxScaler(feature_range=(0, 0.1))

    Feature_scaler.fit(np.array(data))
    Target_scaler.fit(np.array(target).reshape(-1, 1))

    return Feature_scaler, Target_scaler


def scales():
    dataframe = pd.read_csv('NTI_srv_neyronka/management/commands/POWER_SP.csv')
    ALLSKY_SFC_SW_DWN = dataframe['ALLSKY_SFC_SW_DWN']
    remove_row = []
    for i in range(len(ALLSKY_SFC_SW_DWN)):
        if ALLSKY_SFC_SW_DWN[i] < 0:
            remove_row.append(i)

    T2M = dataframe['T2M']
    for i in range(len(T2M)):
        if T2M[i] < -6 and i not in remove_row:
            remove_row.append(i)

    dataframe.drop(index=remove_row, inplace=True)
    dataframe.drop(columns=['LAT', 'LON', 'WS50M', 'DY', 'YEAR', 'MO'], inplace=True)
    Feature_scale, Target_scale = fit_scalers(dataframe)
    return Feature_scale, Target_scale