import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler


# Напишем функцию для проведения препроцессинга полученных данных
def preprocess_data(train_file_path, test_file_path):
    # Загрузим обучающую выборку
    train_df = pd.read_csv(train_file_path)
    # Загрузим тестовую выборку
    test_df = pd.read_csv(test_file_path)

    # Создадим объект класса StandardScaler
    scaler = StandardScaler()

    # Обучим объект класса StandardScaler на обучающей выборке
    scaler.fit(train_df[['temperature']])

    # Применим обученный объект класса StandardScaler к обучающей выборке
    train_scaled_data = scaler.transform(train_df[['temperature']])
    # Применим обученный объект класса StandardScaler к тестовой выборке
    test_scaled_data = scaler.transform(test_df[['temperature']])

    # Сохраним нормализованные тренировочные данные
    train_df['temperature'] = train_scaled_data
    train_df.to_csv(
        train_file_path.replace('.csv', '_preprocessed.csv'), index=False)

    # Сохраним нормализованные тестовые данные
    test_df['temperature'] = test_scaled_data
    test_df.to_csv(
        test_file_path.replace('.csv', '_preprocessed.csv'), index=False)


# Получим количество датасетов
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1

for i in range(n_datasets):
    # Предобработаем и сохраним результаты предобработки тренировочной и тестовой выборок
    preprocess_data(
        f'train/temperature_train_{i+1}.csv',
        f'test/temperature_test_{i+1}.csv')