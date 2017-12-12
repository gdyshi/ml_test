import copy, numpy as np

np.random.seed(0)


#  计算sigmoid非线性函数  非线性变换  输出时使用 该函数多用于回归中
# 前向传播中使用
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative  函数求导
# 用于反向传播中
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# training dataset generation   数值转化     该例子中2进制  8位的加法
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)  # 2**8
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2  # 输入维度  a b
hidden_dim = 16  # 中间层维度
output_dim = 1  # 输出维度  c     c=a+b

# initialize neural network weights 初始化网络权重  -1使其在 【-1，1】之间
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)  # 更新参数的值  权重
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic  训练计算
for j in range(250000):

    # generate a simple addition problem (a + b = c)
    # a_int 随机找一个整数值
    a_int = np.random.randint(largest_number / 2)  # int version
    # 将a_int 转化成2进制数
    a = int2binary[a_int]  # binary encoding
    # 与a同
    b_int = np.random.randint(largest_number / 2)  # int version
    b = int2binary[b_int]  # binary encoding

    # true answer
    #  c_int 为label 值
    c_int = a_int + b_int
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
    # d 为预测值
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
    # 前向传播
    for position in range(binary_dim):
        # generate input and output 输出输入
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)  计算隐藏层的输出 layer_1  是输入层+当前隐藏层权值的递归
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # output layer (new binary representation)  输出层预测  注：输出层权重不递归
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))  # layer_1*w1

        # did we miss?... if so by how much?  
        layer_2_error = y - layer_2
        # 对代价函数求导
        # layer_2_deltas  loss的前向传播
        # layer2_delta = error * 当前层输出值的导数值  ################
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        # 用来打印
        overallError += np.abs(layer_2_error[0])

        # decode estimate so we can print it out  
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep  保存layer_1 用于反向传播隐藏层
        layer_1_values.append(copy.deepcopy(layer_1))
    # 先保存 用于反向传播
    future_layer_1_delta = np.zeros(hidden_dim)

    # 反向传播计算
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])  # 反过来计算回去  位置
        layer_1 = layer_1_values[-position - 1]  # 当前状态值
        prev_layer_1 = layer_1_values[-position - 2]  # 前一状态值 偏1位

        # error at output layer  
        layer_2_delta = layer_2_deltas[-position - 1]  # 赋值给layer_2_delta
        # error at hidden layer
        # layer_1_delta :layer_1层的残差-- 当前循环残差乘以隐藏层权重 + 由前一层残差值乘以当前层值的导数
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + \
                         layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again  权重值的更新
        # w1更新
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        # wh更新
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        # w2更新
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress  
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))

        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))

        print("------------")
