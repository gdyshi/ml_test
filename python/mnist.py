import tensorflow as tf
import numpy as np

layers = [2, 20, 10, 5, 1]
NOISE_LEVEL = 0.15
TRAIN_DATA_NUM = 1000
learning_rate = 0.02


# 产生双月环数据集
def produceData(r, w, d, num):
    r1 = r - w / 2
    r2 = r + w / 2
    # 上半圆
    theta1 = np.random.uniform(0, np.pi, num)
    X_Col1 = np.random.uniform(r1 * np.cos(theta1), r2 * np.cos(theta1), num)[:, np.newaxis]
    X_Row1 = np.random.uniform(r1 * np.sin(theta1), r2 * np.sin(theta1), num)[:, np.newaxis]
    Y_label1 = np.ones(num)  # 类别标签为1

    # 下半圆
    theta2 = np.random.uniform(-np.pi, 0, num)
    X_Col2 = (np.random.uniform(r1 * np.cos(theta2), r2 * np.cos(theta2), num) + r)[:, np.newaxis]
    X_Row2 = (np.random.uniform(r1 * np.sin(theta2), r2 * np.sin(theta2), num) - d)[:, np.newaxis]
    Y_label2 = -np.ones(num)  # 类别标签为-1,注意：由于采取双曲正切函数作为激活函数，类别标签不能为0

    # 合并
    X_Col1 += np.random.normal(0, NOISE_LEVEL, X_Col1.shape)
    X_Row1 += np.random.normal(0, NOISE_LEVEL, X_Row1.shape)
    X_Col2 += np.random.normal(0, NOISE_LEVEL, X_Col2.shape)
    X_Row2 += np.random.normal(0, NOISE_LEVEL, X_Row2.shape)
    X_Col = np.vstack((X_Col1, X_Col2))
    X_Row = np.vstack((X_Row1, X_Row2))
    X = np.hstack((X_Col, X_Row))
    Y_label = np.hstack((Y_label1, Y_label2))
    Y_label.shape = (num * 2, 1)
    return X, Y_label


def set_layers(x):
    for i in range(0, len(layers) - 1):
        X = x if i == 0 else y
        node_in = layers[i]
        node_out = layers[i + 1]
        W = tf.Variable(np.random.randn(node_in, node_out).astype('float32') / (np.sqrt(node_in)))
        b = tf.Variable(np.random.randn(node_out).astype('float32'))
        z = tf.matmul(X, W) + b
        y = tf.nn.tanh(z)
    return y


def main():
    # 产生训练数据
    x_data, y_label = produceData(10, 6, -6, TRAIN_DATA_NUM)
    # 产生测试数据
    x_test, y_test = produceData(10, 6, -6, TRAIN_DATA_NUM)

    x = tf.placeholder(tf.float32, [None, 2])
    y_ = tf.placeholder(tf.float32, [None, 1])


    y = set_layers(x)
    # 均方误差
    mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
    tf.add_to_collection('losses', mse_loss)
    loss = tf.add_n(tf.get_collection('losses'))
    global_step = tf.Variable(0, trainable=False)
    rate = learning_rate

    train_step = tf.train.GradientDescentOptimizer(rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            train_step.run(feed_dict={x: x_data, y_: y_label})
            if i % 1000 == 0:
                train_loss = mse_loss.eval(feed_dict={x: x_data, y_: y_label})
                test_loss = mse_loss.eval(feed_dict={x: x_test, y_: y_test})
                print('step %d, 100*training loss %f,100*testing loss %f' % (i, 100 * train_loss, 100 * test_loss))

    print('DONE!')


if __name__ == '__main__':
    main()
