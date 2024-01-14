# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:29:37 2021

@author: AM4
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

y = df.iloc[0:150, 4].values
# Переводим строковые метки классов в численные значения
class_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
y = np.array([class_mapping[label] for label in y])
# Преобразование вектора меток в матрицу one-hot encoding
y_one_hot = np.eye(3)[y]

X = df.iloc[0:150, 0:4].values

# зададим функцию активации - сигмоида
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

# нам понадобится производная от сигмоиды при вычислении градиента
def derivative_sigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))


# Софтмакс функция активации для многоклассовой классификации
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Производная софтмакса
def derivative_softmax(x):
    return x * (1 - x)


# инициализируем нейронную сеть 
inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 5 # задаем число нейронов скрытого слоя 
outputSize = 3 # количество выходных сигналов равно количеству классов задачи

# веса инициализируем случайными числами, но теперь будем хранить их списком
weights = [
    np.random.uniform(-2, 2, size=(inputSize,hiddenSizes)),  # веса скрытого слоя
    np.random.uniform(-2, 2, size=(hiddenSizes,outputSize))  # веса выходного слоя
]

# прямой проход 
def feed_forward(x):
    input_ = x # входные сигналы
    hidden_ = sigmoid(np.dot(input_, weights[0])) # выход скрытого слоя = сигмоида(входные сигналы*веса скрытого слоя)
    output_ = softmax(np.dot(hidden_, weights[1]))# выход сети (последнего слоя) = сигмоида(выход скрытого слоя*веса выходного слоя)

    # возвращаем все выходы, они нам понадобятся при обратном проходе
    return [input_, hidden_, output_]

# backprop собственной персоной
# на вход принимает скорость обучения, реальные ответы, предсказанные сетью ответы и выходы всех слоев после прямого прохода
def backward(learning_rate, target, net_output, layers):

    # считаем производную ошибки сети
    err = (target - net_output)

    # прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
    # для этого используем chain rule
    # цикл перебирает слои от последнего к первому
    for i in range(len(layers)-1, 0, -1):
        # градиент слоя = ошибка слоя * производную функции активации * на входные сигналы слоя
        
        # ошибка слоя * производную функции активации
        err_delta = err * derivative_softmax(layers[i])   
        
        # пробрасываем ошибку на предыдущий слой
        err = np.dot(err_delta, weights[i - 1].T)
        
        # ошибка слоя * производную функции активации * на входные сигналы слоя
        dw = np.dot(layers[i - 1].T, err_delta)
        
        # обновляем веса слоя
        weights[i - 1] += learning_rate * dw
        
        

# функция обучения чередует прямой и обратный проход
def train(x_values, target, learning_rate):
    output = feed_forward(x_values)
    backward(learning_rate, target, output[2], output)
    return None

def train_stochastic(x_values, target, learning_rate):
    for i in range(len(x_values)):
        random_index = np.random.randint(0, len(x_values))
        x_sample = x_values[random_index].reshape(1, -1)
        y_sample = target[random_index].reshape(1, -1)
        
        output = feed_forward(x_sample)
        backward(learning_rate, y_sample, output[2], output)

# функция предсказания возвращает только выход последнего слоя
def predict(x_values):
    return np.argmax(feed_forward(x_values)[-1], axis=1)


# задаем параметры обучения
iterations = 500
learning_rate = 0.01

# обучаем сеть (фактически сеть это вектор весов weights)
for i in range(iterations):
    train_stochastic(X, y_one_hot, learning_rate)

    if i % 10 == 0:
        accuracy = np.mean(predict(X) == y)
        print("На итерации: " + str(i) + ' || ' + "Точность: " + str(accuracy))

# считаем ошибку на обучающей выборке
pr = predict(X)
print(sum(abs(y-(pr))))


# считаем ошибку на всей выборке
y = df.iloc[:, 4].values
y = np.array([class_mapping[label] for label in y])
X = df.iloc[:, 0:4].values

pr = predict(X)
print(sum(abs(y-(pr))))
print(list(zip(y, pr)))
