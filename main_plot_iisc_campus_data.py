import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import datetime
from sklearn import preprocessing

def normalize(df):
    result = df.copy()
    for feature_name in ['Flow']:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

if __name__ == "__main__":

    data = pd.read_csv('data/rev1.csv', delimiter=';', decimal=',')
    columns_to_drop = ['Unnamed: 0', 'MonthFlow', 'Serial', 'EpochTime', 'DatenTime']#, 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'DatenTime', 'EpochTime', 'Serial']
    data = data.drop(columns=columns_to_drop, axis=1)
    #data = data.iloc()[0:20000]
    data['FM NO.'] = data['FM NO.'].astype('category')
    data['Flow'] = pd.to_numeric(data['Flow'], downcast="float")
    data['PT'] = pd.to_numeric(data['PT'], downcast="float")
    sensors = data['FM NO.'].unique()

    #converted_dates = list(map(datetime.datetime.strptime, data['Date']+'-'+data['Time'], len(data['Time'])*['%d/%m/%Y-%H:%M:%S']))
    #converted_dates = list(map(datetime.datetime.strptime, data['DatenTime'], len(data['DatenTime'])*['%Y-%m-%d %H:%M:%S']))
    #data['Normaltime']
    #data['DateTime'] = converted_dates
    #data = data.drop(columns=['Date', 'Time'], axis=1)
    data['Normaltime'] = pd.to_datetime(data['Normaltime'], format='%Y - %m - %d %H : %M : %S')
    data = data.sort_values(by='Normaltime')
    data = data.set_index('Normaltime')
    #data = normalize(data)
    plt.figure(figsize=(10, 10))
    for sensor in ['FM15']:
        sensor_data = data[data['FM NO.'] == sensor]
        #sensor_data = sensor_data[sensor_data['DateTime'][0].date()==datetime.date(2022, 8, 29)]
        plt.plot(sensor_data.index, sensor_data['PT'], label=sensor + ' PT')
        plt.plot(sensor_data.index, sensor_data['Flow'], '--', label=sensor + ' Flow')
    plt.legend()
    plt.xticks(rotation = 45)
    #plt.grid()
    plt.show()




