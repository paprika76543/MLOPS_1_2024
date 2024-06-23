import numpy as np
import matplotlib.pyplot as plt

# Генерация 24 точек с нормальным распределением
mean = 0  # Среднее значение
std_dev = 1  # Стандартное отклонение

points = np.random.normal(mean, std_dev, 24)

# Визуализация данных
plt.figure(figsize=(10, 6))
plt.plot(np.arange(24), points, marker='o', linestyle='-')
plt.xlabel('Часы')
plt.ylabel('Значения')
plt.title('Генерация 24 точек с нормальным распределением')
plt.grid(True)
plt.show()