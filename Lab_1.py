# import matplotlib.pyplot as plt
# x = [1, 2, 3, 4, 5]
# y = [1, 4, 9, 16, 25]
# plt.plot(x ,y)
# plt.show()

import numpy as np         
import pandas as pd       
from scipy import stats 

# библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')    # стиль графиков
# %matplotlib inline

norm_rv = stats.norm(loc=0, scale=1)  # генератор нормального распределения

sample = norm_rv.rvs(1000, random_state=1)  # сгенерируем 1000 значений
print(sample[:10])

norm_rv.pdf(sample)

# получение значений плотности для элементов выборки, понадобится для построения графиков
x = sample
1 / np.sqrt(2 * np.pi) * np.exp(-(x ** 2) / 2)

norm_rv.pdf(1)

x = np.linspace(-3, 3, 100)
print(x)
pdf = norm_rv.pdf(x)
pdf

plt.plot(x, pdf)
# plt.ylabel('$f(x)$')
# plt.xlabel('$x$')

# На ней же нарисуем f(1)
plt.scatter([-1, 1, 2], [norm_rv.pdf(-1),norm_rv.pdf(1), norm_rv.pdf(2)], color="blue")
# plt.show()

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# На ней же нарисуем f(1)
plt.scatter([0,1,2], [norm_rv.pdf(0), norm_rv.pdf(1), norm_rv.pdf(2)], color="blue")

norm_rv.cdf(1)

x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# На ней же нарисуем f(1)
plt.scatter([1], [norm_rv.pdf(1)], color="blue")

# на ту же картинку добавили новую часть, штриховку
xq = np.linspace(-3, 1, 100)
yq = norm_rv.pdf(xq)
plt.fill_between(xq, 0, yq, color='blue', alpha=0.2)

plt.axvline(1, color='blue', linestyle="--", lw=2)

x = np.linspace(-3, 3, 100)
cdf = norm_rv.cdf(x)

plt.plot(x, cdf)
plt.ylabel('$F(x)$')
plt.xlabel('$x$')

# На ней же нарисуем F(1)
plt.scatter([1], [norm_rv.cdf(1)], color="blue")

norm_rv.cdf(3) - norm_rv.cdf(1)

x = np.linspace(-5, 5, 100)
pdf = norm_rv.pdf(x)

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# На ней же нарисуем f(1)
plt.scatter([1, 3], [norm_rv.pdf(1), norm_rv.pdf(3)], color="blue")

# на ту же картинку добавили новую часть, штриховку
xq = np.linspace(1, 3)
yq = norm_rv.pdf(xq)
plt.fill_between(xq, 0, yq, color='blue', alpha=0.2)

plt.axvline(1, color='blue', linestyle="--", lw=2)
plt.axvline(3, color='blue', linestyle="--", lw=2)

# plt.show()

# здесь использовано правило трёх сигм. Для своих выборок можете использовать например функцию np.quantile()
uroven = (1 - 0.6826)/2
q = norm_rv.ppf(uroven)
q

x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)

plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

xq = np.linspace(-3, q)
yq = norm_rv.pdf(xq)
plt.fill_between(xq, 0, yq, color='blue', alpha=0.2)

plt.axvline(q, color='blue', linestyle="--", lw=2)

y_max = plt.ylim()[1]
plt.text(q + 0.1, 0.8*y_max, round(q,2), color='blue', fontsize=16)
# plt.show()

# эмпирическое распределение
sample[:10]
sample.shape
np.mean(sample)  # выборочное среднее

x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)

# плотность 
plt.plot(x, pdf, lw=3)

# гистограмма, параметр density отнормировал её. 
plt.hist(sample, bins=100, density=True)

plt.ylabel('$f(x)$')
plt.xlabel('$x$')

# для построения ECDF используем библиотеку statsmodels
from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF(sample)   # строим эмпирическую функцию по выборке

plt.step(ecdf.x, ecdf.y)
plt.ylabel('$F(x)$', fontsize=20)
plt.xlabel('$x$', fontsize=20)
x = np.linspace(-3, 3, 100)

# теоретическа cdf 
cdf = norm_rv.cdf(x)
plt.plot(x, cdf, label='theoretical CDF')

# эмпирическая сdf
ecdf = ECDF(sample)
plt.step(ecdf.x, ecdf.y, label='empirical CDF')

plt.ylabel('$F(x)$')
plt.xlabel('$x$')
plt.legend(loc='upper left')

plt.show()
