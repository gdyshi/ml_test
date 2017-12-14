import numpy as np


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


batch_size = 1

x_data = np.linspace(-1, 1, batch_size)
# y_data = x_data * 0.2 + 0.3 + np.random.normal(0, 0.55, x_data.size)
y_data = x_data * 0.2 + 0.3

x_dim = 1
y_dim = 1
learning_rate = 1
# X = np.ones((x_dim, batch_size))
# Y = np.ones((y_dim, batch_size))
X = x_data.reshape((x_dim, batch_size))
Y = y_data.reshape((y_dim, batch_size))

l0_dim = x_dim
l1_dim = 1

w1 = np.random.random((l0_dim, l1_dim))
# print('w1:' + str(w1.shape))
b1 = np.zeros((l1_dim, 1))
# print('b1:' + str(b1.shape))

for i in range(100):
    print('i:' + str(i))

    A0 = X
    # print('a0:' + str(a0.shape))
    # def forward:
    Z1 = np.add(np.dot(w1.transpose(), A0), b1)
    # print('z1:' + str(z1.shape))
    A1 = sigmoid(Z1)
    # print('a1:' + str(a1.shape))
    print('w1:'+str(w1)+'A0:'+str(A0)+'b1:'+str(b1)+'Z1:'+str(Z1))
    print('dot:'+str(np.dot(w1.transpose(), A0)))

    Y_ = A1

    cost = (Y_ - Y) ** 2 / 2
    loss = cost.sum() / batch_size
    # print('X:'+str(X)+'Y:'+str(Y)+'Y_:'+str(Y_)+'loss:'+str(loss))
    print('X:' + str(X))
    print('Y:' + str(Y))
    print('Y_:' + str(Y_))
    print('loss:' + str(loss))

    # def backward:
    d_cost = (Y_ - Y) / batch_size
    # print('d_cost:' + str(d_cost.shape))
    # d_loss_cost = 1 / batch_size

    d_A1_Z1 = sigmoid_output_to_derivative(Z1)
    # print('d_a1_z1:' + str(d_a1_z1.shape))
    d_Z1 = d_cost * d_A1_Z1
    # print('d_Z1:' + str(d_Z1.shape))
    d_Z1_b1 = np.ones_like(b1)
    # print('d_Z1_b1:' + str(d_Z1_b1.shape))
    d_b1 = d_Z1_b1 * d_Z1.sum()
    # print('d_b1:' + str(d_b1.shape))
    d_Z1_w1 = A0
    # print('d_z1_w1:' + str(d_z1_w1.shape))
    d_w1 = np.dot(d_Z1_w1, d_Z1.transpose())
    # print('d_w1:' + str(d_w1.shape))
    d_Z1_A0 = w1
    # print('d_z1_a0:' + str(d_z1_a0.shape))
    d_A0 = np.dot(d_Z1_A0, d_Z1)
    # print('d_a0:' + str(d_a0.shape))
    # print('A1:'+str(A1))
    print('w1:' + str(w1) + 'b1:' + str(b1))
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1
    print('d_w1:' + str(d_w1) + 'd_b1:' + str(d_b1))
    print('w1:' + str(w1) + 'b1:' + str(b1))
