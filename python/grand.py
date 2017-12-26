import numpy as np


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


batch_size = 500
# batch_size = 1
learning_rate = 2

x_dim = 1
y_dim = 1
l0_dim = x_dim
# l1_dim = 20
l1_dim = 5
l2_dim = y_dim

ebsilon = 0.01


def forward(A0, Y, w1, b1, w2, b2):
    Z1 = np.add(np.dot(w1.transpose(), A0), b1)
    A1 = sigmoid(Z1)
    Z2 = np.add(np.dot(w2.transpose(), A1), b2)
    A2 = sigmoid(Z2)
    Y_ = A2
    cost = (Y_ - Y) ** 2 / 2
    loss = cost.sum() / batch_size
    return Z1, A1, Z2, A2, Y_, loss


def backward(Y, Y_, A0, A1, A2, Z1, Z2, w1, w2, b1, b2):
    d_cost = (Y_ - Y) / batch_size
    d_A2 = d_cost

    d_A2_Z2 = sigmoid_output_to_derivative(A2)
    d_Z2 = d_A2 * d_A2_Z2
    d_Z2_b2 = np.ones_like(b2)
    d_b2 = d_Z2_b2 * d_Z2.sum()
    d_Z2_w2 = A1
    d_w2 = np.dot(d_Z2_w2, d_Z2.transpose())
    d_Z2_A1 = w2
    d_A1 = np.dot(d_Z2_A1, d_Z2)

    d_A1_Z1 = sigmoid_output_to_derivative(A1)
    # d_A1_Z1 = sigmoid_output_to_derivative(Z1)
    d_Z1 = d_A1 * d_A1_Z1
    d_Z1_b1 = np.ones_like(b1)
    d_b1 = d_Z1_b1 * d_Z1.sum()
    d_Z1_w1 = A0
    d_w1 = np.dot(d_Z1_w1, d_Z1.transpose())
    d_Z1_A0 = w1
    d_A0 = np.dot(d_Z1_A0, d_Z1)
    return d_w1, d_w2, d_b1, d_b2


def init(batch_size):
    x_data = np.linspace(-1, 1, batch_size)
    y_data = x_data * 0.2 + 0.3
    X = x_data.reshape((x_dim, batch_size))
    Y = y_data.reshape((y_dim, batch_size))

    w1 = np.random.random((l0_dim, l1_dim))
    b1 = np.zeros((l1_dim, 1))
    w2 = np.random.random((l1_dim, l2_dim))
    b2 = np.zeros((l2_dim, 1))

    return X, Y, w1, b1, w2, b2


def grand_check():
    X, Y, w1, b1, w2, b2 = init(batch_size)
    A0 = X
    Z1, A1, Z2, A2, Y_, loss = forward(A0, Y, w1, b1, w2, b2)
    d_w1, d_w2, d_b1, d_b2 = backward(Y, Y_, A0, A1, A2, Z1, Z2, w1, w2, b1, b2)
    sita = []
    sita_approx = []
    for i in range(l0_dim):
        for j in range(l1_dim):
            w1[i][j] -= ebsilon
            _, _, _, _, _, loss_i = forward(A0, Y, w1, b1, w2, b2)
            w1[i][j] += ebsilon * 2
            _, _, _, _, _, loss_x = forward(A0, Y, w1, b1, w2, b2)
            w1[i][j] -= ebsilon
            d_approx = (loss_x - loss_i) / (2 * ebsilon)
            sita_approx.append(d_approx)
            sita.append(d_w1[i][j])
    for i in range(l1_dim):
        b1[i] -= ebsilon
        _, _, _, _, _, loss_i = forward(A0, Y, w1, b1, w2, b2)
        b1[i] += ebsilon * 2
        _, _, _, _, _, loss_x = forward(A0, Y, w1, b1, w2, b2)
        b1[i] -= ebsilon
        d_approx = (loss_x - loss_i) / (2 * ebsilon)
        sita_approx.append(d_approx)
        sita.append(d_b1[i])
    for i in range(l1_dim):
        for j in range(l2_dim):
            w2[i][j] -= ebsilon
            _, _, _, _, _, loss_i = forward(A0, Y, w1, b1, w2, b2)
            w2[i][j] += ebsilon * 2
            _, _, _, _, _, loss_x = forward(A0, Y, w1, b1, w2, b2)
            w2[i][j] -= ebsilon
            d_approx = (loss_x - loss_i) / (2 * ebsilon)
            sita_approx.append(d_approx)
            sita.append(d_w2[i][j])
    for i in range(l2_dim):
        b2[i] -= ebsilon
        _, _, _, _, _, loss_i = forward(A0, Y, w1, b1, w2, b2)
        b2[i] += ebsilon * 2
        _, _, _, _, _, loss_x = forward(A0, Y, w1, b1, w2, b2)
        b2[i] -= ebsilon
        d_approx = (loss_x - loss_i) / (2 * ebsilon)
        sita_approx.append(d_approx)
        sita.append(d_b2[i])

    sita_approx = np.array(sita_approx)
    sita = np.array(sita)
    # w10
    check_value = np.sum((sita_approx - sita) * (sita_approx - sita)) / (
            np.sum(sita_approx * sita_approx) + np.sum(sita * sita))
    print(sita_approx)
    print(sita)
    print(check_value)
    # print(d_approx)
    # print(grand)
    # print(check_value)


def train():
    X, Y, w1, b1, w2, b2 = init(batch_size)
    A0 = X
    for i in range(10000):
        Z1, A1, Z2, A2, Y_, loss = forward(A0, Y, w1, b1, w2, b2)
        d_w1, d_w2, d_b1, d_b2 = backward(Y, Y_, A0, A1, A2, Z1, Z2, w1, w2, b1, b2)
        w2 -= learning_rate * d_w2
        b2 -= learning_rate * d_b2
        w1 -= learning_rate * d_w1
        b1 -= learning_rate * d_b1
        if 0 == i % 500:
            print(loss)


# train()
grand_check()

