import numpy as np


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)

x_data = np.linspace(-100, 100, 10000)
# y_data = x_data * 0.2 + 0.3 + np.random.normal(0, 0.55, x_data.size)
y_data = x_data * 0.2 + 0.3

x_dim = 1
y_dim = 1
learning_rate = 0.5
x = np.ones((x_dim, 1))
y = np.ones((y_dim, 1))

l0_dim = x_dim
l1_dim = 10
l2_dim = 1

w1 = np.random.random((l0_dim, l1_dim))
# print('w1:' + str(w1.shape))
b1 = np.zeros((l1_dim, 1))
# print('b1:' + str(b1.shape))
w2 = np.random.random((l1_dim, l2_dim))
# print('w2:' + str(w2.shape))
b2 = np.zeros((l2_dim, 1))
# print('b2:' + str(b2.shape))

for i in range(10):
    print('i:'+ str(i))
    x[0][0] = (1)*0.01
    y[0][0] = (x[0][0])* 0.2 + 0.3
    a0 = x
    # print('a0:' + str(a0.shape))
    # def forward:
    z1 = np.add(np.dot(w1.transpose(), a0), b1)
    # print('z1:' + str(z1.shape))
    a1 = sigmoid(z1)
    # print('a1:' + str(a1.shape))

    z2 = np.add(np.dot(w2.transpose(), a1), b2)
    # print('z2:' + str(z2.shape))
    a2 = sigmoid(z2)
    # print('a2:' + str(z2.shape))

    y_ = a2

    cost = (y_ - y) ** 2 / 2
    # loss = cost.sum() / batch_size
    print('x:'+str(x)+'y:'+str(y)+'y_:'+str(y_)+'cost:'+str(cost))

    # def backward:
    # d_loss_cost = 1 / batch_size
    # print('d_loss_cost:' + str(d_loss_cost.shape))
    d_cost_a2 = (y_ - y)
    # print('d_cost_a3:' + str(d_cost_a3.shape))
    d_cost = d_cost_a2
    # print('d_cost:' + str(d_cost.shape))

    d_a2_z2 = sigmoid_output_to_derivative(z2)
    # print('d_a2_z2:' + str(d_a2_z2.shape))
    d_z2 = d_cost * d_a2_z2
    # print('d_z2:' + str(d_z2.shape))
    d_z2_b2 = np.ones_like(b2)
    # print('d_z2_b2:' + str(d_z2_b2.shape))
    d_b2 = d_z2_b2 * d_z2
    # print('d_b2:' + str(d_b2.shape))
    d_z2_w2 = a1
    # print('d_z2_w2:' + str(d_z2_w2.shape))
    d_w2 = np.dot(d_z2_w2, d_z2.transpose())
    # print('d_w2:' + str(d_w2.shape))
    d_z2_a1 = w2
    # print('d_z2_a1:' + str(d_z2_a1.shape))
    d_a1 = np.dot(d_z2_a1, d_z2)
    # print('d_a1:' + str(d_a1.shape))
    print('w2:'+str(w2))
    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2

    d_a1_z1 = sigmoid_output_to_derivative(z1)
    # print('d_a1_z1:' + str(d_a1_z1.shape))
    d_z1 = d_a1 * d_a1_z1
    # print('d_z1:' + str(d_z1.shape))
    d_z1_b1 = np.ones_like(b1)
    # print('d_z1_b1:' + str(d_z1_b1.shape))
    d_b1 = d_z1_b1 * d_z1
    # print('d_b1:' + str(d_b1.shape))
    d_z1_w1 = a0
    # print('d_z1_w1:' + str(d_z1_w1.shape))
    d_w1 = np.dot(d_z1_w1, d_z1.transpose())
    # print('d_w1:' + str(d_w1.shape))
    d_z1_a0 = w1
    # print('d_z1_a0:' + str(d_z1_a0.shape))
    d_a0 = np.dot(d_z1_a0, d_z1)
    # print('d_a0:' + str(d_a0.shape))
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1
    print('a1:'+str(a1))
    print('d_w2:'+str(d_w2))
