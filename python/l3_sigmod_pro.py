import numpy as np


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


batch_size = 500
learning_rate = 2

x_dim = 1
y_dim = 1
l0_dim = x_dim
l1_dim = 20
l2_dim = 5
l3_dim = y_dim
layers = [x_dim, 20, 5, y_dim]

x_data = np.linspace(-1, 1, batch_size)
y_data = x_data * 0.2 + 0.3
X = x_data.reshape((x_dim, batch_size))
Y = y_data.reshape((y_dim, batch_size))
print('X:' + str(X))
print('Y:' + str(Y))

W = []
B = []

for i in range(0, len(layers) - 1):
    if i == 0:
        w = 0
        b = 0
    else:
        w = np.random.random((layers[i - 1], layers[i]))
        b = np.zeros((layers[i], 1))
    W.append(w)
    B.append(b)

for i in range(10000):
    # def forward:
    Z = []
    A = []
    for i in range(0, len(layers) - 1):
        if i == 0:
            z = 0
            a = X
        else:
            z = np.add(np.dot(W[i].transpose(), A[i-1]), b[i])
            a = sigmoid(z)
        Z.append(z)
        A.append(a)

    Y_ = a
    cost = (Y_ - Y) ** 2 / 2
    loss = cost.sum() / batch_size

    # def backward:
    d_cost = (Y_ - Y) / batch_size
    d_A3 = d_cost

    d_A3_Z3 = sigmoid_output_to_derivative(A3)
    d_Z3 = d_A3 * d_A3_Z3
    d_Z3_b3 = np.ones_like(b3)
    d_b3 = d_Z3_b3 * d_Z3.sum()
    d_Z3_w3 = A2
    d_w3 = np.dot(d_Z3_w3, d_Z3.transpose())
    d_Z3_A2 = w3
    d_A2 = np.dot(d_Z3_A2, d_Z3)
    w3 -= learning_rate * d_w3
    b3 -= learning_rate * d_b3

    d_A2_Z2 = sigmoid_output_to_derivative(A2)
    d_Z2 = d_A2 * d_A2_Z2
    d_Z2_b2 = np.ones_like(b2)
    d_b2 = d_Z2_b2 * d_Z2.sum()
    d_Z2_w2 = A1
    d_w2 = np.dot(d_Z2_w2, d_Z2.transpose())
    d_Z2_A1 = w2
    d_A1 = np.dot(d_Z2_A1, d_Z2)
    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2

    d_A1_Z1 = sigmoid_output_to_derivative(A1)
    d_Z1 = d_A1 * d_A1_Z1
    d_Z1_b1 = np.ones_like(b1)
    d_b1 = d_Z1_b1 * d_Z1.sum()
    d_Z1_w1 = A0
    d_w1 = np.dot(d_Z1_w1, d_Z1.transpose())
    d_Z1_A0 = w1
    d_A0 = np.dot(d_Z1_A0, d_Z1)
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1
    if (0 == i % 100):
        print('i:' + str(i) + '  loss:' + str(loss))

print('Y_:' + str(Y_))
