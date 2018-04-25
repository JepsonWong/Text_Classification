#encoding:utf-8

import tensorflow as tf
import os
import time
from datetime import timedelta
from sklearn import metrics
import numpy as np
# import tensorflow.contrib.keras as kr
import keras as kr

import os,sys
sys.path.append("../../")

from feature_engineering.preprocess.get_news_subset import get_news_subset


# CNN网络的配置
class SimpleTextCNN_Config(object):
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度

    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表大小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

class SimpleTextCNN(object):
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def cnn(self):
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self, x_train, y_train, x_val, y_val):
        data_len = len(x_train)
        indices = np.random.permutation(np.arange(data_len))
        x_train = x_train[indices]
        y_train = y_train[indices]

        print("Configuring TensorBoard and Saver...")
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        tensorboard_dir = 'tensorboard/simple_text_cnn'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)

        # 配置 Saver
        save_dir = 'checkpoints/simple_text_cnn'
        save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

        saver = tf.train.Saver()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Loading training and validation data...")
        # 载入训练集与验证集, x_train, y_train, x_val, y_val

        # 创建session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

        flag = False
        for epoch in range(self.config.num_epochs):
            print('Epoch:', epoch + 1)
            epoch_size = (len(x_train) - 1) / self.config.batch_size + 1
            data_len = len(x_train)
            for i in range(epoch_size):
                start_id = i * self.config.batch_size
                end_id = min((i + 1) * self.config.batch_size, data_len)
                feed_dict = {
                    self.input_x: x_train[(start_id):(end_id)],
                    self.input_y: y_train[(start_id):(end_id)],
                    self.keep_prob: self.config.dropout_keep_prob
                }

                session.run(self.optim, feed_dict=feed_dict)  # 运行优化

                if total_batch % self.config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % self.config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    loss_train, acc_train = session.run([self.loss, self.acc], feed_dict=feed_dict)
                    feed_dict[self.keep_prob] = 1.0
                    feed_dict[self.input_x] = x_val
                    feed_dict[self.input_y] = y_val
                    loss_val, acc_val = session.run([self.loss, self.acc], feed_dict=feed_dict)

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    time_dif = self.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                break

    def test(self, x_test, y_test):
        print("Loading test data...")
        start_time = time.time()

        # 配置 Saver
        save_dir = 'checkpoints/simple_text_cnn'
        save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

        print('Testing...')
        feed_dict = {}
        feed_dict[self.keep_prob] = 1.0
        feed_dict[self.input_x] = x_test
        feed_dict[self.input_y] = y_test
        feed_dict[self.input_y] = y_test
        loss_test, acc_test = session.run([self.loss, self.acc], feed_dict=feed_dict)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))

        y_test_cls = np.argmax(y_test, 1)
        y_pred_cls = session.run(self.y_pred_cls, feed_dict=feed_dict) # 保存预测结果

        # 评估
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test_cls, y_pred_cls))

        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)

        time_dif = self.get_time_dif(start_time)
        print("Time usage:", time_dif)

class TextCNN_Config(object):
    seq_length = 600 # 序列长度
    num_classes = 10  # 类别数
    vocab_size = 5000  # 词汇表大小

    # 模型参数
    embedding_dim = 128 # 词向量维度
    num_filters = 128  # 卷积核数目
    kernel_size = [3, 4, 5]  # 卷积核尺寸
    # hidden_dim = 128  # 全连接层神经元
    dropout_keep_prob = 0.5  # dropout保留比例
    l2_reg_lambda = 0.0 # L2 regularization lambda (default: 0.0)

    learning_rate = 1e-3  # 学习率
    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
        self.cnn()

    def cnn(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        # 对于3、4、5的卷积核尺寸，分别创建一个卷积层和池化层
        pooled_outputs = []
        for i, filter_size in enumerate(config.kernel_size):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                # shape: [batch_size, seq_legnth, embedding_dim, 1]*[filter_size, embedding_size, 1, num_filters] = [batch_size, seq_legnth-filter_size+1, 1, num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # shape: [batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = self.config.num_filters * len(self.config.kernel_size)
        # shape: [batch_size, 1, 1, num_filters*3]
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape: [batch_size, num_filters*3]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # 这个函数返回值是一个向量，不是一个值，所以需要tf.reduce_mean对向量里面的所有元素求和
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def train(self, x_train, y_train, x_val, y_val):
        data_len = len(x_train)
        indices = np.random.permutation(np.arange(data_len))
        x_train = x_train[indices]
        y_train = y_train[indices]

        # 创建session
        sess = tf.Session()

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join('tensorboard/text_cnn', "train")
        if not os.path.exists(train_summary_dir):
            os.makedirs(train_summary_dir)
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join('tensorboard/text_cnn', "dev")
        if not os.path.exists(dev_summary_dir):
            os.makedirs(dev_summary_dir)
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # 前面是保存结果的文件夹，后面是保存结果
        checkpoint_dir = os.path.abspath(os.path.join("checkpoints/text_cnn"))
        save_path = os.path.join("checkpoints/text_cnn", "best_validation")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        best_acc_val = 0.0  # 最佳验证集准确率

        for epoch in range(self.config.num_epochs):
            print('Epoch:', epoch + 1)
            epoch_size = (len(x_train) - 1) / self.config.batch_size + 1
            data_len = len(x_train)
            for i in range(epoch_size):
                start_id = i * self.config.batch_size
                end_id = min((i + 1) * self.config.batch_size, data_len)
                feed_dict = {
                    self.input_x: x_train[(start_id):(end_id)],
                    self.input_y: y_train[(start_id):(end_id)],
                    self.dropout_keep_prob: self.config.dropout_keep_prob
                }

                _, current_step = sess.run([train_op, global_step], feed_dict=feed_dict)  # 运行优化

                if current_step % self.config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    current_step1, train_summary = sess.run([global_step, train_summary_op], feed_dict=feed_dict)
                    train_summary_writer.add_summary(train_summary, current_step1)

                if current_step % self.config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    loss_train, acc_train = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    feed_dict[self.dropout_keep_prob] = 1.0
                    feed_dict[self.input_x] = x_val
                    feed_dict[self.input_y] = y_val
                    loss_val, acc_val, dev_summary, current_step2 = sess.run([self.loss, self.accuracy, dev_summary_op, global_step], feed_dict=feed_dict)
                    dev_summary_writer.add_summary(dev_summary, current_step2)

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        saver.save(sess=sess, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    time_dif = self.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(current_step, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

    def test(self, x_train, y_train, x_val, y_val):
        pass


if __name__ == '__main__':
    train_news = get_news_subset("../../data/THUCNews的一个子集/cnews.train.txt", "utf-8")
    train_news.set_stopwords('../../data/stop_words_zh.utf8.txt')
    x_train, y_train = train_news.get_character_ids_and_labels() # 必须在定义val_news之前，因为val_news要用train_news.words和train_news.characters，而这些在这个函数里才生成
    y_train_pad = kr.utils.to_categorical(y_train, num_classes=10)
    #test_news = get_news_subset("../../data/THUCNews的一个子集/cnews.test.txt", "utf-8", words=train_news.words, characters=train_news.characters)
    val_news = get_news_subset("../../data/THUCNews的一个子集/cnews.val.txt", "utf-8", words=train_news.words, characters=train_news.characters)
    #test_news.set_stopwords('../../data/stop_words_zh.utf8.txt')
    val_news.set_stopwords('../../data/stop_words_zh.utf8.txt')

    #x_test, y_test = test_news.get_character_ids_and_labels()
    x_val, y_val = val_news.get_character_ids_and_labels()
    y_val_pad = kr.utils.to_categorical(y_val, num_classes=10)
    '''
    y_val_pad = []
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in y_val:
        if i + 1 <= 9:
            y_val_pad.append(x[0:i] + [1] + x[i + 1:])
        else:
            y_val_pad.append(x[0:i] + [1])
    y_val_pad = np.array(y_val_pad)
    x_val = np.array(x_val)
    '''

    print('Configuring CNN model...')

    '''
    config = SimpleTextCNN_Config()
    model = SimpleTextCNN(config)
    model.train(x_train, y_train_pad, x_val, y_val_pad)
    '''

    config = TextCNN_Config()
    model = TextCNN(config)
    model.train(x_train, y_train_pad, x_val, y_val_pad)
