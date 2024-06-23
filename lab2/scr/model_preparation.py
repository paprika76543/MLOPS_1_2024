from sklearn.linear_model import LinearRegression       # Модель линейной регрессии
import pandas as pd                                     # Библиотека Pandas для работы с табличными данными
from joblib import dump                                 # в scikit-learn ничего такого особенного нет
                                                        # пользуемся joblib

# Читаем данные из файла
DF = pd.read_csv('lab2/train/data_train_prep.csv', delimiter=',', header=0, index_col=False)

# Выделяем признаки и целевой показатель
X_train, y_train = DF.iloc[:,0:1], DF.iloc[:,1]
#print(X_train)

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Метрики
r_sq = model.score(X_train, y_train)
print('Coefficient of determination', r_sq * 100, '%')
print('intercept_', model.intercept_)
print('Coefficients', model.coef_)

# Сохраняем обученную модель в файл
dump(model, 'lab2/model/model.joblib')  # чтобы сохранить объект

print("Модель записана в файл model.joblib")