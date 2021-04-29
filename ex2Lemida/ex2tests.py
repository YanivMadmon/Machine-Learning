import numpy as np
import sys
import random


def knn(x_train, x_test, y_train, k):
    x_train, x_test = zscore(x_train, x_test)
    distance = [0] * x_train.shape[0]
    result_all = []
    for test in x_test:
        i = 0
        for x in x_train:
            distance[i] = np.linalg.norm(test-x)
            i += 1
        dis = np.array(distance.copy())
        result = [0] * k
        for j in range(k):
            index = np.where(dis == dis.min())
            result[j] = y_train[index][0]
            dis[index] = np.max(dis) + 1
        result_all.append(func_result(result))
    acc = func_miss(y_train, result_all)
    return acc


def func_miss(y_train, result):
    len_result = result.__len__()
    currect = 0
    for i in range(len_result):
        if result[-i] == y_train[-i]:
            currect += 1
    return (currect/len_result)*100

def func_result(result):
    final = [0] * 3
    for r in result:
        if r == 0:
            final[0] += 1
        if r == 1:
            final[1] += 1
        if r == 2:
            final[2] += 1
    if final[0] > final[1] and final[0] > final[2]:
        return 0
    elif final[1] > final[0] and final[1] > final[2]:
        return 1
    elif final[2] > final[0] and final[2] > final[1]:
        return 2
    elif final[0] == final[1]:
        return random.choice([0, 1])
    elif final[0] == final[2]:
        return random.choice([0, 2])
    elif final[1] == final[2]:
        return random.choice([1, 2])

    return random.choice([0, 1, 2])



def minmax(x_train, x_test):
    for i in range(12):
        min = np.min(x_train, 0)[i]
        max = np.max(x_train, 0)[i]
        for x in x_train:
            x[i] = (x[i] - min) / (max - min)
        for x in x_test:
            x[i] = (x[i] - min) / (max - min)
    return x_train, x_test


def zscore(x_train, x_test):
    for i in range(12):
        mean = np.mean(x_train, 0)[i]
        std = np.std(x_train, 0)[i]
        for x in x_train:
            x[i] = (x[i] - mean) / std
        for x in x_test:
            x[i] = (x[i] - mean) / std
    return x_train, x_test


def perceptron(x_train, x_test, y_train):
    w = np.random.rand(3, 12)
    b = np.array([0.0]*3)
    eta = 0.01
    epochs = 11
    x_train, x_test = zscore(x_train, x_test)
    for e in range(epochs):
        i = 0
        for x in x_train:
            y_hat = np.argmax(np.dot(w, x) + b)
            y = int(y_train[i])
            if y_train[i] != y_hat:
                w[y] = w[y] + eta * x
                b[y] = b[y] + eta
                w[y_hat] = w[y_hat] - eta * x
                b[y_hat] = b[y_hat] - eta
            i += 1
    result_all = per_test(w, x_test, b)
    acc = func_miss(y_train, result_all)
    return acc


def per_test(w, x_test, b):
    result_all = []
    for x in x_test:
        y_hat = np.argmax(np.dot(w, x) + b)
        result_all.append(y_hat)
    return result_all


def pa(x_train, x_test, y_train):
    w = np.random.rand(3, 12)
    b = np.array([0.0] * 3)
    epochs = 10
    x_train, x_test = zscore(x_train, x_test)
    for e in range(epochs):
        i = 0
        for x in x_train:
            y = int(y_train[i])
            w_y = w[y]
            w_new = w.copy()
            w_new[y] = [0]*12
            y_hat = np.argmax(np.dot(w_new, x) + b)
            w_y_hat = w[y_hat]

            tau = max(0, (1.0 - np.dot(w_y, x) + np.dot(w_y_hat, x))) / (2 * ((np.linalg.norm(x)) ** 2))
            w[y] = w[y] + tau * x
            b[y] = b[y] + tau
            w[y_hat] = w[y_hat] - tau * x
            b[y_hat] = b[y_hat] - tau
            i += 1
    result_all = per_test(w, x_test, b)
    acc = func_miss(y_train, result_all)
    return acc


def make_array(file):
    x = []
    i = 0
    with open(file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        x.append(line.split(","))
        x[i] = [w.replace('W', '1.0') for w in x[i]]
        x[i] = [w.replace('R', '0.0') for w in x[i]]
        i += 1
    x = np.array(x.copy())
    x = x.astype(np.float)
    return x


def split_array(x_train):

    len = x_train.shape[0]
    n_90 = int(0.9 * len)
    return x_train[:n_90], x_train[n_90:]


def shuffle(first_set, second_set):
    assert len(first_set) == len(second_set)
    permutation = np.random.permutation(len(first_set))
    return first_set[permutation], second_set[permutation]


def main():
    train_x,  train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]

    x_train = make_array(train_x)
    y_train = np.loadtxt(train_y)
    x_test = make_array(test_x)

    x_train, y_train = shuffle(x_train, y_train)

    av = 0
    k = 11

    for i in range(10):
        x_train, y_train = shuffle(x_train, y_train)
        x_set, x_train_test = split_array(x_train)

        m = perceptron(x_set, x_train_test, y_train)
        av += m
        print(f"knn = 7  round {i+1} accuracy: {m}")
    av = av/10
    print(f"knn = 7 average accuracy: {av}")


    #output = open(f"output.txt", "w")
    return

if __name__ == "__main__":
    main()
