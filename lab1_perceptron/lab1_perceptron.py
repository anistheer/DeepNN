# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# загружаем и подготавляваем данные
df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи


# создаем матрицу весов скрытого слоя
Win = np.zeros((1+inputSize,hiddenSizes)) 
# пороги w0 задаем случайными числами
Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes))) 
# остальные веса  задаем случайно -1, 0 или 1 
Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes))) 

#Wout = np.zeros((1+hiddenSizes,outputSize))

# случайно инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)
   
# функция прямого прохода (предсказания) 
def predict(Xp):
    # выходы первого слоя = входные сигналы * веса первого слоя
    hidden_predict = np.where((np.dot(Xp, Win[1:,:]) + Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
    # выходы второго слоя = выходы первого слоя * веса второго слоя
    out = np.where((np.dot(hidden_predict, Wout[1:,:]) + Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict


# обучение
# у перцептрона Розенблатта обучаются только веса выходного слоя 
# как и раньше обучаем подавая по одному примеру и корректируем веса в случае ошибки
n_iter = 10000
eta = 0.1
n_last_steps = 20  # количество последних шагов для отслеживания средней ошибки
error_threshold = 0.1  # порог для средней ошибки
max_consecutive_cycles = 20

errors_history = []
consecutive_cycles = 0

for i in range(n_iter):
    errors = []
    
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr, hidden = predict(xi) 
        errors.append(np.abs(target - pr) ** 2)
        
        Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
        Wout[0] += eta * (target - pr)
    
    mean_error = np.mean(errors)
    errors_history.append(mean_error)

    print(mean_error, errors_history);

    if True:
        avg_last_errors = np.mean(errors_history[-n_last_steps:])
        if mean_error < error_threshold:
            print(f"Модель сошлась на {i} шаге")
            print("Wout:", Wout.reshape(1, -1))
            print("Win:", Win.reshape(1, -1), "\n")
            break

        if mean_error >= errors_history[-1]:
            consecutive_cycles += 1
        else:
            consecutive_cycles = 0

        if consecutive_cycles >= max_consecutive_cycles:
            print(f"Достигнуто максимальное число последовательных циклов без улучшения ({max_consecutive_cycles}). Обучение прервано.")
            break

# посчитаем сколько ошибок делаем на всей выборке
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
pr, hidden = predict(X)

print("zip(y, pr)", list(zip(y, pr.flatten())))
print("Result:", sum(pr-y.reshape(-1, 1)))

# далее оформляем все это в виде отдельного класса neural.py
