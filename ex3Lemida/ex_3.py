# 204293005 Yaniv Madmon

import numpy as np
import sys

sigmoid = lambda x: 1 / (1 + np.exp(-x))

h_l = 90


def shuffle(first_set, second_set):
    assert len(first_set) == len(second_set)
    permutation = np.random.permutation(len(first_set))
    return first_set[permutation], second_set[permutation]


def make_array(file):
    x = []
    i = 0
    with open(file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        x.append(line.split(" "))
        i += 1
    x = np.array(x.copy())
    x = x.astype(np.float)
    return x


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def fprop(x, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1, x).reshape(h_l, 1) + b1

    h1 = sigmoid(z1)

    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)

    ret = {'x': x, 'y_hat': np.argmax(h2), 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache, y):
    # Follows procedure given in notes
    x, z1, h1, z2, h2 = [fprop_cache[key] for key in ('x', 'z1', 'h1', 'z2', 'h2')]

    # the correct y
    c_y = np.zeros((10, 1))
    c_y[y] = 1.0

    #  dL/dz2
    dz2 = (h2 - c_y)

    #  dL/dz2 * dz2/dw2
    dW2 = np.dot(dz2, h1.T)

    #  dL/dz2 * dz2/db2
    db2 = dz2
    #  dL/dz2 * dz2/dh1 * dh1/dz1
    dz1 = np.dot(fprop_cache['w2'].T,
                 (h2 - c_y)) * sigmoid(z1) * (1 - sigmoid(z1))
    #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    dW1 = np.dot(dz1, x.reshape(784, 1).T)
    db1 = dz1
    return {'b1': db1, 'w1': dW1, 'b2': db2, 'w2': dW2}


def test(params, x_test):
    file_ans = open("test_y", 'w')

    for i in range(len(x_test)):
        x = x_test[i]
        fprop_cache = fprop(x, params)
        answer = str(fprop_cache['y_hat'])
        file_ans.write(answer + '\n')
    file_ans.close()


def main():
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]

    x_train_all = make_array(train_x)
    # normalization
    x_train = x_train_all / 255
    y_train_all = np.loadtxt(train_y).astype(int)
    y_train = y_train_all

    x_test = make_array(test_x)
    # normalization
    x_test = x_test / 255

    # Initialize random parameters and inputs
    w1 = np.random.uniform(-0.05, 0.05, [h_l, 784])
    b1 = np.random.uniform(-0.05, 0.05, [h_l, 1])
    w2 = np.random.uniform(-0.05, 0.05, [10, h_l])
    b2 = np.random.uniform(-0.05, 0.05, [10, 1])
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    n = 0.007
    epochs = 30

    for e in range(epochs):
        i = 0
        x_train, y_train = shuffle(x_train, y_train)
        for x in x_train:

            fprop_cache = fprop(x, params)
            bprop_cache = bprop(fprop_cache, y_train[i])

            # update w and b
            params['w1'] = params['w1'] - np.dot(bprop_cache['w1'], n)
            params['b1'] = params['b1'] - np.dot(bprop_cache['b1'], n)
            params['w2'] = params['w2'] - np.dot(bprop_cache['w2'], n)
            params['b2'] = params['b2'] - np.dot(bprop_cache['b2'], n)

            i += 1
        print(e)
    test(params, x_test)


if __name__ == "__main__":
    main()
