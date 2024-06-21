import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import os
import sys


# Функция для обучения модели и вычисления метрики
def train_model_and_evaluate(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Перемешивание данных

    df = shuffle(df, random_state=42)

    # Разделим данные на признаки и целевую переменную
    X = df[['temperature']]
    y = df['anomaly']

    # Создадим модель логистической регрессии
    model = LogisticRegression()

    # Обучим модель
    model.fit(X, y)

    # Выполним предикт
    y_pred = model.predict(X)

    # Вычислим метрику
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Результаты оценки работы модели соберем в дата фрейм
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1]
    })

    return model, results


# Получим количество дата сетов
if len(sys.argv) > 1:
    n_datasets = int(sys.argv[1])
else:
    n_datasets = 1

# Создадим директорию для хранения модели
os.makedirs('models', exist_ok=True)
print()
for i in range(n_datasets):
    # Обучим модель на предобработанных данных
    model, results = train_model_and_evaluate(
        f'train/temperature_train_{i+1}_preprocessed.csv')

    # Сохраним обученную модель
    joblib.dump(model, f'models/model_{i+1}.pkl')

    print(f"The model for the data set {i+1} is trained.")
    print(results.to_string(index=False))
    print('-' * 20)