import numpy as np
import tensorflow as tf
import os

print(os.getcwd())

# 指定本地路径
local_path = 'mnist_data\\mnist.npz'

# 使用 numpy 加载数据集
with np.load(local_path, allow_pickle=True) as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']


# 超参数
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# 神经网络参数
n_hidden_1 = 256 # 第一层神经元个数
n_hidden_2 = 256 # 第二层神经元个数
num_input = 784 # MNIST 输入数据(图像大小: 28*28)
num_classes = 10 # MNIST 手写体数字类别 (0-9)

# 输入到数据流图中的训练数据
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# 权重和偏置
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
    # 第一层隐藏层（256个神经元）
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # 第二层隐藏层（256个神经元）
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # 输出层
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# 构建模型
logits = neural_net(X)

# 定义损失函数和优化器
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# 定义预测准确率
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有变量（赋默认值）
init = tf.global_variables_initializer()

mnist = tf.data.Dataset.from_tensor_slices((x_train, y_train))
mnist = mnist.batch(batch_size)
iterator = mnist.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:

    # 执行初始化操作
    sess.run(init)
    sess.run(iterator.initializer)
    
    for step in range(1, num_steps+1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y = sess.run(next_element)
        # batch_x, batch_y = mnist.next_batch(batch_size)
        # 执行训练操作，包括前向和后向传播
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # 计算损失值和准确率
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")