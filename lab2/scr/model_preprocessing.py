import pandas as pd                                 # Библиотека Pandas для работы с табличными данными
from sklearn.preprocessing import StandardScaler    # Импортируем стандартизацию от scikit-learn
from sklearn.pipeline import Pipeline               # Пайплайн


# Загрузим данные из файла "data_train.csv"
DF_train = pd.read_csv('lab2/train/data_train.csv')

# Выполним предобработку данных с помощью sklearn.preprocessing.StandardScaler
scaler = StandardScaler()

# Обучаем на тренировочных данных
X_train = DF_train['x']

# Преобразование Series в DataFrame с одним столбцом
data_frame = X_train.to_frame()

# Стандартизируем данные
X_train_prep = scaler.fit_transform(data_frame)

# Запишем данные в файл
DF_train['x'] = X_train_prep
#print(DF_train)
DF_train.to_csv(f'lab2/train/data_train_prep.csv', index=False)
print("Стандартизованные тренировочные данные записаны в файл data_train_prep.csv")

# Загрузим данные из файла "data_test.csv"
DF_test = pd.read_csv('lab2/test/data_test.csv')

# Обучаем на тренировочных данных
X_test = DF_test['x']

#print(X_train)
# Выполним предобработку данных с помощью sklearn.preprocessing.StandardScaler
scaler = StandardScaler()

# Преобразование Series в DataFrame с одним столбцом
data_frame = X_test.to_frame()

# Стандартизируем данные
X_test_prep = scaler.fit_transform(data_frame)
#print(X_train_prep)

# Запишем данные в файл
DF_test['x'] = X_test_prep
print(DF_test)
DF_test.to_csv(f'lab2/test/data_test_prep.csv', index=False)
print("Стандартизованные тестовые данные записаны в файл data_train_prep.csv")