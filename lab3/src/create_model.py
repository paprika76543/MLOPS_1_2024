import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump                                 # в scikit-learn ничего такого особенного нет
                                                        # пользуемся joblib

# Получаем правильные пути
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))      # Каталог со скриптами
PROJECT_PATH = os.path.dirname(SCRIPTS_PATH)                   # Каталог проекта

# Загрузка датасета iris
iris = load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

print(X_test[:10])
# Обучение модели случайного леса
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Предсказание на тестовом наборе
rf_y_pred = rf_model.predict(X_test)

# Оценка точности модели случайного леса
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Сохраняем обученную модель в файл
path_to_file = os.path.join(PROJECT_PATH, 'model', 'model.joblib')

try:
    dump(rf_model, path_to_file)
    print('Модель успешно сохранена в:', path_to_file)
except Exception as e:
    print('Произошла ошибка при сохранении модели:', e)