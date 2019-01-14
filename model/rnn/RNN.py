#encoding:utf-8

import tensorflow as tf

# LSTM网络的配置
class Config(object):
    """
    num_layers为LSTM网络的层数
    num_steps为LSTM展开的步数。
    hidden_size为隐层单元数目，每个词会表示成[hidden_size]大小的向量，当然也不一定单词成都必须为[hidden_size]。
    max_epoch：当epoch < max_epoch时，lr_decay值=1；epoch > max_epoch时，lr_decay逐渐减小。
    max_max_epoch为epoch次数。
    keep_prob为LSTM输出层keep的概率，如果小于1说明做了drop的优化
    batch_size为一个batch的数据个数。
    vocab_size为单词个数
    """
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2 #
    num_steps = 20 #
    hidden_size = 200 #
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0 #
    lr_decay = 0.5
    batch_size = 20 #
    vocab_size = 10000 #
    num_labels = 20 #

class LSTM(object):
    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.keep_prob = config.keep_prob
        self.num_layers = config.num_layers
        self.num_labels = config.num_labels

        self.content = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.label = tf.placeholder(tf.int32, [self.batch_size, 1])

        '''
        使用BasicLSTMCell构建一个基础LSTM单元，然后根据keep_prob来为cell配置dropout。最后通过MultiRNNCell将num_layers个lstm_cell连接起来。
        在LSTM单元中，有2个状态值，分别是c和h。
        '''
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.dropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(self.batch_size, tf.float32)

        '''
        embedding部分 如果为训练状态，后面加一层Dropout
        '''
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.vocab_size, self.hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.content)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # 定义输出output，lstm的中间变量复用
        # LSTM循环
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        print "cell_output shape: ", cell_output.shape
        print "state shape: ", len(state)
        print "state[0] type: ", type(state[0])
        print "state[1] type: ", type(state[1])
        print "state[0]: ", (state[0])
        print "state[1]: ", (state[1])

        # 得到网络最后的输出
        # 把之前的list展开，成[batch, hidden_size*num_steps]，然后 reshape，成[batch*num_steps, hidden_size]。
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.num_labels], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.num_labels], dtype=tf.float32)
        # softmax_w shape = [hidden_size, num_labels]
        # logits shape = [batch*num_steps, num_labels]
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 带权重的交叉熵计算
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.label, [-1])],
                                                                  [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)])
        self._cost = tf.reduce_sum(loss) / self.batch_size
        self._final_state = state
        if not is_training:
            return

    def train(self, x_train, y_train, x_val, y_val):
        pass

    def test(self, x_test, y_test):
        pass

