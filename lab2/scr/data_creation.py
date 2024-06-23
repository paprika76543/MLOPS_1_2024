import numpy as np                                      # библиотека Numpy для операций линейной алгебры и прочего
import pandas as pd                                     # Библиотека Pandas для работы с табличными данными
from sklearn.model_selection import train_test_split    # функция разбиения на тренировочную и тестовую выборку

def true_fun(x, a=np.pi, b = 0, f=np.sin):
    """Генерация произвольной зависимости
    Входные переменные:
    ===========
    x: массив данных из которых будет генерироваться зависимость
    a: коэффициент на который входные данные будут умножаться
    если а - список, то это коэффициенты в полиномиальной зависимости
    так, а = [1,2,3] позволит сгенерировать зависимость вида 1*x+ 2*x^2 +3*x^3
    b: коэффициент который будет добавлен к данным (постоянная прибавочка)
    f: функция которая будет применена к зависимости. Можно задать списком. Тогда это будут разные колонки
    """
    x = np.atleast_1d(x)[:] # убеждаемся что данные - одномерный массив
    a = np.atleast_1d(a)

    if f is None: f = lambda x:x # если функция не задана (None) то ничего не происходит
    x = np.sum([ai*np.power(x, i+1) for i,ai in enumerate(a)],axis=0) # домнажаем входные данные на коэффициенты (и если надо возводим в степень)

    return f(x+ b)

def noises(shape , noise_power):
    """Генерация случайного шума
    Входные переменные:
    ===========
    shape: размерность массива данных
    noise_power: коэффициент ~ сила шума
    """
    return np.random.randn(*shape) *noise_power # библиотека numpy может генерировать случайные числа.
                                                # в данном случае - нормальное распределение (среднее = 0, стандартное отклонение = 1 )

def dataset(a, b, f = None,  N = 250, x_max =1, noise_power = 0, random_x = True,  seed = 42):
    """Генерация набора данных
    Входные переменные:
    ===========
    a: коэффициент на который входные данные будут умножаться
    если а - список, то это коэффициенты в полиномиальной зависимости
    так, а = [1,2,3] позволит сгенерировать зависимость вида 1*x+ 2*x^2 +3*x^3
    b: коэффициент который будет добавлен к данным (постоянная прибавочка)
    f: функция которая будет применена к зависимости. Можно задать списком. Тогда это будут разные колонки
    N: количество точек данных
    x_max: максимальное значение данных
    noise_power: коэффициент ~ сила шума
    random_x: как будут распределены данные (линейно, или случайно)
    seed: фиксированный сид случайных чисел (для повторяемости)
    """

    np.random.seed(seed) # фиксируем случайный seed

    if random_x:# если мы хотим случайно распределить данные
        x = np.sort(np.random.rand(N))*x_max # то x будет N случайных числе из диапазона от 0 до x_max
    else: # иначе
        x = np.linspace(0,x_max,N) # х это равномерно распределенные N чисел из диапазона от 0 до x_max

    y_true = np.array([]) # создаем пустой массив который будет "наполнять" зависимостями

    for f_ in np.append([], f): # если f - задана списком, то мы учтем все варианты
        y_true=np.append(y_true, true_fun(x, a, b, f_)) # применяем описанную выше функцию true_fun

    #y_true = y_true.reshape(-1,N).T
    y = y_true + noises(y_true.shape , noise_power) # добавляем шум

    return y, y_true, x #np.atleast_2d(x).T # возвращаем зашумленные значения зависимостей, зависимости без шума, и массив входных данных

# Тренировочная и тестовая выборка
# Получаем зашумленные значения зависимостей, зависимости без шума, и массив входных данных
y, y_true, X = dataset(a = 3, b = 8,
                       f = None,  N = 200,
                       x_max =30,
                       noise_power = 0.1,
                       seed = 42)

# print(y, y_true, X)

# Сохраним данные с шумами
# разбиваем данные на тренировочные и тестовые
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
# print(x_train.shape, y_train.shape,  x_test.shape, y_test.shape )
# print(x_test)

# Создание DataFrame из массивов
df_train = pd.DataFrame({'x': x_train, 'y': y_train})
# print(type(x_train))
# print(x_train)
# print(type(y_train))
# print(y_train)
df_train.to_csv(f'lab2/train/data_train.csv', index=False)

df_test = pd.DataFrame({'x': x_test, 'y': y_test})
df_test.to_csv(f'lab2/test/data_test.csv', index=False)
print("Записаны данные с шумами в файлы data_train.csv, data_test.csv")

# Сохраним данные без шумов
# разбиваем данные на тренировочные и тестовые
x_train_true, x_test_true, y_train_true, y_test_true = train_test_split(X, y_true,
                                                    test_size=0.3,
                                                    random_state=42)
# Создание DataFrame из массивов
df_train_true = pd.DataFrame({'x': x_train_true, 'y': y_train_true})
df_test_true = pd.DataFrame({'x': x_test_true, 'y': y_test_true})

df_train_true.to_csv(f'lab2/train/data_train_true.csv', index=False)
df_test_true.to_csv(f'lab2/test/data_test_true.csv', index=False)

print("Записаны данные без шумов в файлы data_train_true, data_test_true.csv")