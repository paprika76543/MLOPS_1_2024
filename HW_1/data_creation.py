import numpy as np
import pandas as pd
import os
import sys

# Создаем директории для хранения наших сгенерированных данных
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)


# Напишем непосредственно функцию генерирования данных температуры
def generate_data(n_samples,
                  anomaly_ratio=0.1,
                  anomaly_loc=30,
                  anomaly_scale=10):

    # Генерируем данные без аномалий
    data = np.random.normal(loc=20, scale=5, size=(n_samples, 1))

    # Подсчитаем количество экземпляров данных с аномалиями
    n_anomalies = int(n_samples * anomaly_ratio)

    # Сгенерируем экземпляры данных с аномалиями и добавим их к нашим сгенерированным данным
    anomalies = np.random.normal(loc=anomaly_loc, scale=anomaly_scale,
                                 size=(n_anomalies, 1))
    data = np.concatenate((data, anomalies), axis=0)

    # Округлим полученные данные до одного знака после запятой (точки)
    data = np.round(data, 1)

    # Создадим лейблы
    labels = np.zeros(data.size, dtype=int)
    labels[n_samples:] = 1  # Аномалиям присвоим лейбл 1

    #  Структурируем полученные данные
    dtype = [('data', np.float32), ('labels', np.int32)]
    data_with_labels = np.empty(data.size, dtype=dtype)
    data_with_labels['data'] = data.flatten()
    data_with_labels['labels'] = labels


    data_dict = {'temperature': [temp for temp, anomaly in data_with_labels],
                 'anomaly': [anomaly for temp, anomaly in data_with_labels]}

    return data_dict


# Создадим датасеты
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1  # Default value if no argument is passed

for i in range(n_datasets):
    # Создадим обучающую выборку и поместим ее в соответствующую директорию
    train_data = generate_data(1000,
                               anomaly_ratio=0.1,
                               anomaly_loc=30+i*5,
                               anomaly_scale=10+i*2)
    df_train = pd.DataFrame(train_data)
    df_train.to_csv(f'train/temperature_train_{i+1}.csv', index=False)

    # Создадим тестовую выборку и поместим ее в соответствующую директорию
    test_data = generate_data(200,
                              anomaly_ratio=0.1,
                              anomaly_loc=30+i*5,
                              anomaly_scale=10+i*2)
    df_test = pd.DataFrame(test_data)
    df_test.to_csv(f'test/temperature_test_{i+1}.csv', index=False)