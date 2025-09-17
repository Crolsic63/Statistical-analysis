import numpy as np         
import pandas as pd       
from scipy import stats 

# библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')    # стиль графиков

# Параметры F-распределения (степени свободы)
dfn = 5  # степени свободы числителя
dfd = 10  # степени свободы знаменателя

# Генератор F-распределения
f_rv = stats.f(dfn=dfn, dfd=dfd)

# Сгенерируем 1000 значений из F-распределения
sample = f_rv.rvs(1000, random_state=1)
print("Первые 10 значений выборки:")
print(sample[:10])

# Получение значений плотности для элементов выборки
pdf_values = f_rv.pdf(sample)

# Создаем диапазон значений для построения графика
# F-распределение определено от 0 до бесконечности
x = np.linspace(0, 5, 100)  # обычно смотрим до 5, так как хвост быстро убывает
pdf = f_rv.pdf(x)

# Построение графика плотности распределения
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label=f'F({dfn}, {dfd}) распределение', linewidth=2)
plt.ylabel('$f(x)$', fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.title(f'Плотность F-распределения с параметрами ({dfn}, {dfd})', fontsize=16)

# Отметим несколько характерных точек
characteristic_points = [0.5, 1.0, 2.0]
plt.scatter(characteristic_points, f_rv.pdf(characteristic_points), 
           color="red", s=100, zorder=5, label='Точки плотности')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Построение графика функции распределения (CDF)
plt.figure(figsize=(10, 6))
cdf = f_rv.cdf(x)
plt.plot(x, cdf, label=f'F({dfn}, {dfd}) CDF', linewidth=2)
plt.ylabel('$F(x)$', fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.title(f'Функция распределения F({dfn}, {dfd})', fontsize=16)

# Отметим несколько характерных точек
plt.scatter(characteristic_points, f_rv.cdf(characteristic_points), 
           color="red", s=100, zorder=5, label='Точки CDF')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Построение гистограммы выборки и теоретической плотности
plt.figure(figsize=(10, 6))
plt.hist(sample, bins=50, density=True, alpha=0.7, 
         label='Гистограмма выборки', color='lightblue')

# Теоретическая плотность
plt.plot(x, pdf, 'r-', lw=2, label='Теоретическая плотность')

plt.ylabel('Плотность', fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.title(f'Сравнение выборки и теоретического F({dfn}, {dfd}) распределения', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Построение эмпирической и теоретической функций распределения
from statsmodels.distributions.empirical_distribution import ECDF

plt.figure(figsize=(10, 6))

# Теоретическая CDF
plt.plot(x, cdf, label='Теоретическая CDF', linewidth=2)

# Эмпирическая CDF
ecdf = ECDF(sample)
plt.step(ecdf.x, ecdf.y, label='Эмпирическая CDF', linewidth=2)

plt.ylabel('$F(x)$', fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.title(f'Сравнение теоретической и эмпирической CDF F({dfn}, {dfd})', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Вывод основных характеристик
print(f"\nОсновные характеристики F({dfn}, {dfd}) распределения:")
print(f"Математическое ожидание: {f_rv.mean():.4f}")
print(f"Дисперсия: {f_rv.var():.4f}")
print(f"Медиана: {f_rv.median():.4f}")
print(f"Мода: {f_rv.mode()[0]:.4f}") if hasattr(f_rv.mode(), '__len__') else print(f"Мода: {f_rv.mode():.4f}")

# Квантили распределения
quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
print(f"\nКвантили распределения:")
for q in quantiles:
    print(f"F({q:.2f}) = {f_rv.ppf(q):.4f}")