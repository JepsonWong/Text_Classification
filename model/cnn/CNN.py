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
class Config(object):
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度

    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

class CNN(object):
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
        #data_len = len(x_train)
        #indices = np.random.permutation(np.arange(data_len))
        #x_train = x_train[indices]
        #y_train = y_train[indices]

        print("Configuring TensorBoard and Saver...")
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        tensorboard_dir = 'tensorboard/cnn'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)

        # 配置 Saver
        save_dir = 'checkpoints/cnn'
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
            epoch_size = len(x_train) / self.config.batch_size
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
        save_dir = 'checkpoints/cnn'
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

if __name__ == '__main__':
    train_news = get_news_subset("../../data/THUCNews的一个子集/cnews.train.txt", "utf-8")
    #test_news = get_news_subset("../../data/THUCNews的一个子集/cnews.test.txt", "utf-8")
    val_news = get_news_subset("../../data/THUCNews的一个子集/cnews.val.txt", "utf-8", words=train_news.words, characters=train_news.characters)
    train_news.set_stopwords('../../data/stop_words_zh.utf8.txt')
    #test_news.set_stopwords('../../data/stop_words_zh.utf8.txt')
    val_news.set_stopwords('../../data/stop_words_zh.utf8.txt')

    x_train, y_train = train_news.get_character_ids_and_labels()
    y_train_pad = kr.utils.to_categorical(y_train, num_classes=10)
    '''
    y_train_pad= []
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in y_train:
        if i + 1 <= 9:
            y_train_pad.append(x[0:i] + [1] + x[i + 1:])
        else:
            y_train_pad.append(x[0:i] + [1])
    y_train_pad = np.array(y_train_pad)
    
    x_train = np.array(x_train)
    '''
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
    config = Config()

    model = CNN(config)
    model.train(x_train, y_train_pad, x_val, y_val_pad)
