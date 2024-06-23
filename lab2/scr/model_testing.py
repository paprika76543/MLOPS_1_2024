from sklearn.linear_model import LinearRegression   # Модель линейной регрессии
from sklearn.metrics import mean_squared_error
import pandas as pd                                 # Библиотека Pandas для работы с табличными данными
from joblib import load                       # в scikit-learn ничего такого особенного нет, пользуемся joblib


# Загрузим обученную модель
model = load('lab2/model/model.joblib')

# Загрузим данные из файла "data_test.csv"
DF = pd.read_csv('lab2/test/data_test_prep.csv')

# Разделим данные на факторы (X) и целевую переменную (y):
# X = data_test['x'].values.reshape(-1, 1)
# y = data_test['y'].values
X, y = DF.iloc[:,0:1], DF.iloc[:,1]

# Сделаем предсказания на основе обученной модели
predictions = model.predict(X)

# Оценим качество модели, например, посчитав среднеквадратичную ошибку:
mse = mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)