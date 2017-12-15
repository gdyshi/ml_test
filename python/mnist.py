import numpy as np


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


x_dim = 2
y_dim = 1
batch_size = 100
learning_rate = 0.1
X = np.ones((x_dim, batch_size))
Y = np.ones((y_dim, batch_size))

l0_dim = x_dim
l1_dim = 10
l2_dim = 5
l3_dim = 1

A0 = X.reshape((l0_dim, batch_size))
print('A0:' + str(A0.shape))
# def forward:
W1 = np.random.random((l0_dim, l1_dim))
print('W1:' + str(W1.shape))
B1 = np.zeros((l1_dim, 1))
print('B1:' + str(B1.shape))
Z1 = np.add(np.dot(W1.transpose(), A0), B1)
print('Z1:' + str(Z1.shape))
A1 = sigmoid(Z1)
print('A1:' + str(A1.shape))

W2 = np.random.random((l1_dim, l2_dim))
print('W2:' + str(W2.shape))
B2 = np.zeros((l2_dim, 1))
print('B2:' + str(B2.shape))
Z2 = np.add(np.dot(W2.transpose(), A1), B2)
print('Z2:' + str(Z2.shape))
A2 = sigmoid(Z2)
print('A2:' + str(Z2.shape))

W3 = np.random.random((l2_dim, l3_dim))
print('W3:' + str(W3.shape))
B3 = np.zeros((l3_dim, 1))
print('B3:' + str(B3.shape))
Z3 = np.add(np.dot(W3.transpose(), A2), B3)
print('Z3:' + str(Z3.shape))
A3 = sigmoid(Z3)
print('A3:' + str(A3.shape))

Y_ = A3

cost = (Y_ - Y) ** 2 / 2
loss = cost.sum() / batch_size
print(loss)

# def backward:
d_loss_cost = 1 / batch_size
# print('d_loss_cost:' + str(d_loss_cost.shape))
d_cost_A3 = (Y_ - Y)
print('d_cost_A3:' + str(d_cost_A3.shape))
d_cost = d_loss_cost * d_cost_A3
print('d_cost:' + str(d_cost.shape))

d_A3_Z3 = sigmoid_output_to_derivative(Z3)
print('d_A3_Z3:' + str(d_A3_Z3.shape))
d_Z3 = d_cost * d_A3_Z3
print('d_Z3:' + str(d_Z3.shape))
d_Z3_W3 = A2
print('d_Z3_W3:' + str(d_Z3_W3.shape))
d_W3 = d_Z3*d_Z3_W3
print('d_W3:' + str(d_W3.shape))
d_Z3_B3 = np.ones((l3_dim, batch_size))
# print('d_Z3_B3:' + str(d_Z3_B3.shape))
d_B3 = np.dot(d_Z3_B3, d_Z3.transpose())
d_B3 = d_Z3*d_Z3_B3
print('d_B3:' + str(d_B3.shape))
d_Z3_A2 = W3
print('d_Z3_A2:' + str(d_Z3_A2.shape))
d_A2 = np.dot(d_Z3_A2, d_Z3)
print('d_A2:' + str(d_A2.shape))

W3 -= learning_rate * d_W3
B3 -= learning_rate * d_B3

d_A2_Z2 = sigmoid_output_to_derivative(Z2)
print('d_A2_Z2:' + str(d_A2_Z2.shape))
d_Z2 = d_A2 * d_A2_Z2
print('d_Z2:' + str(d_Z2.shape))
d_Z2_W2 = A1
print('d_Z2_W2:' + str(d_Z2_W2.shape))
d_W2 = np.dot(d_Z2_W2, d_Z2.transpose())
print('d_W2:' + str(d_W2.shape))
d_Z2_B2 = np.ones((l2_dim, batch_size))
# print('d_Z2_B2:' + str(d_A3_Z3.shape))
d_B2 = np.dot(d_Z2_B2, d_Z2.transpose())
print('d_B2:' + str(d_B2.shape))
W2 -= learning_rate * d_W2
B2 -= learning_rate * d_B2

