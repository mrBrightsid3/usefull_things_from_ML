"""ВАЖНОЕ
OvA = OvR one versus all/rest позволяет расширять любой двоичный классификатор с целью решения многклассовых задач
обучаем по одному классификатору на класс, при этом специфический класс трактуется как
положительный, а образцы из всех остальных классов считаются
принадлежащими отрицательным классам."""
import numpy as np
import pandas as pd


class Perceptron(object):
    """Классификатор на основе персептрона.
    Параметры
    eta : float
    Скорость обучения (между О . О и 1 . 0 )
    п iter : int
    Пр оходы по обучающему набору данных .
    random state : int
    Начальное значение генератора случайных чисел
    для инициализации случайными весами.
    Атрибуты
    w_ : одномерный массив
    Веса после подгонки .
    errors : список
    Количество неправильных классификаций (обно влений) в каждой эпохе ."""

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Подгоняет к обучающим данным .
        Параметры
        Х : {подобен массиву} , форма = [n_examples , n_features]
        Обучающие векторы , где n_examples - количество образцов
        и n_features - количеств о признаков .
        у : подобен массиву , форма = [n_examples]
        Целевые значения .
        Возвращает
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # ско = 0.01
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """вычисляем общий вход"""
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv("iris.csv")
print(df.tail())
