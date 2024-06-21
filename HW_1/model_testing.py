import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import sys


# Напишем функцию для тестирования ранее обученной модели
def test_model(model_path, test_data_path):
    # Загрузим ранее обученную модель
    model = joblib.load(model_path)

    # Загрузим тестовую выборку
    df_test = pd.read_csv(test_data_path)

    # Также выделим признаки и целевую переменную
    X_test = df_test[['temperature']]
    y_test = df_test['anomaly']

    # Выполним предикт
    y_pred = model.predict(X_test)

    # Вычислим метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Соберем результаты вычисления метрик в дата фрейм
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return results


# Получим количество дата сетов
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1
print()
for i in range(n_datasets):
    # Укажем путь к ранее предобученной модели
    model_path = f'models/model_{i+1}.pkl'
    # Укажем путь к тестовой выборке
    test_data_path = f'test/temperature_test_{i+1}_preprocessed.csv'

    # Запустим тестирование модели
    results = test_model(model_path, test_data_path)
    print(f"The model for dataset {i+1} is tested.")
    print(results.to_string(index=False))
    print('-' * 20)