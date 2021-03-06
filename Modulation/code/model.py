# -*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from sklearn.metrics import confusion_matrix

def one_hot(value, dim):
    """
    value must be a list. values[i] is from 1 to dim.

    """
    if hasattr(value, '__len__'):
        val = np.zeros([len(value), dim])
        for i, v in enumerate(value):
            val[i, v - 1] = 1
        return val
    else:
        val = np.zeros([dim])
        val[value - 1] = 1
        return val


class Config(object):
    """
        configs for model
    """

    # 1Dconv configs
    conv1_filter_sizes = [50]
    conv1_filter_num = 32
    conv1_padding = 'VALID'
    conv2_filter_sizes = [40]
    conv2_filter_num = 16
    conv2_padding = 'VALID'

    # pooling configs
    conv1_pool_sizes = [10]
    conv2_pool_sizes = [10]

    # params
    l2_reg_lambda = 1e-2
    learning_rate = 1e-3

    # others
    batch_size = 64
    epochs = 20
    sequence_length = 10240
    sequence_width = 1
    label_size = 7

    # config
    batch_show = 20
    model_save_step = 5

class SignalModModel(object):
    def __init__(self, config, log_path, model_path):
        self.config = config
        self.add_placeholders()
        self.outputs = self.add_model(self.inputs)
        self.pred = tf.argmax(tf.nn.softmax(self.outputs), axis=1)
        self.loss = self.add_loss_op(self.outputs)
        self.accu = self.add_accu_op(self.outputs)
        self.train_op = self.add_train_op(self.loss)

        # add summary op
        tf.summary.scalar('accuracy', self.accu)
        tf.summary.scalar('total_loss', self.loss)

        self.log_path = log_path
        self.merged_summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_path,
                            graph=tf.get_default_graph())

        # model saver
        self.model_path = model_path
        self.saver = tf.train.Saver()

        # gpu config
        cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # 只占用20%的GPU内存
        cf.gpu_options.per_process_gpu_memory_fraction = 0.2
        cf.gpu_options.visible_device_list = '0'

        # session
        self.sess = tf.Session(config=cf)
        self.sess.run(tf.global_variables_initializer())


    def load_test_data(self, test_datapath):
        with open(test_datapath, 'rb') as fr:
            x = list()
            y = list()
            for i, line in enumerate(fr):
                items = line.strip().split('\t')
                signal = map(float, items[:-1])
                label = int(items[-1])
                label = one_hot(label, self.config.label_size)
                x.append(signal)
                y.append(label)
            return x, y


    def data_iterator(self, data_path):
        """

        Data format: 10240 signal time points + signal mixture type

        :param data_path: file path where data is read
        :return: a batch of input data
        """
        with open(data_path, 'rb') as fr:
            batch_x = list()
            batch_y = list()
            for i, line in enumerate(fr):
                items = line.strip().split('\t')
                signal = map(float, items[:-1])
                label = int(items[-1])
                label = one_hot(label, self.config.label_size)
                batch_x.append(signal)
                batch_y.append(label)
                if (i + 1) % self.config.batch_size == 0:
                    yield batch_x, batch_y
                    del batch_x
                    del batch_y
                    batch_x = list()
                    batch_y = list()
            if len(batch_x) != 0 and len(batch_y) != 0:
                yield batch_x, batch_y


    # 输入
    def add_placeholders(self):
        self.inputs = tf.placeholder(np.float32,
                shape=[None, self.config.sequence_length],
                name='signal')
        self.labels = tf.placeholder(np.int32,
                shape=[None, self.config.label_size],
                name='mix_type')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    # CNN层
    def add_model(self, inputs):
        outputs = list()
        inputs = tf.reshape(inputs, [-1, self.config.sequence_length,
                                     self.config.sequence_width, 1])
        for i, filter_size in enumerate(self.config.conv1_filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                # 第一层的filter的W和b
                conv1_W = tf.get_variable('conv1_W',
                    shape=[filter_size, self.config.sequence_width, 1, self.config.conv1_filter_num],
                    initializer=tf.truncated_normal_initializer(.0, .1))
                conv1_b = tf.get_variable('conv1_b',
                    initializer=tf.constant(0.1, shape=[self.config.conv1_filter_num]))
                # 卷积
                conv1_out = tf.nn.relu(
                    (tf.nn.conv2d(
                    inputs, conv1_W, [1, 1, 1, 1], padding=self.config.conv1_padding) + conv1_b))
                # 池化
                pool1_b = tf.get_variable('pool1_b',
                    initializer=tf.constant(0.1, shape=[self.config.conv1_filter_num]))
                pool1_out = tf.nn.max_pool(conv1_out,
                    [1, self.config.conv1_pool_sizes[i], 1, 1], [1, self.config.conv1_pool_sizes[i], 1, 1],
                    padding=self.config.conv1_padding)
                pool1_out = tf.nn.tanh(pool1_out + pool1_b)

                # 第一层的filter的W和b
                conv2_W = tf.get_variable('conv2_W',
                    shape=[self.config.conv2_filter_sizes[i], pool1_out.get_shape()[2], pool1_out.get_shape()[3], self.config.conv2_filter_num],
                    initializer=tf.truncated_normal_initializer(.0, .1))
                conv2_b = tf.get_variable('conv2_b',
                    initializer=tf.constant(0.1, shape=[self.config.conv2_filter_num]))
                # 卷积
                conv2_out = tf.nn.relu(
                    (tf.nn.conv2d(
                    pool1_out, conv2_W, [1, 1, 1, 1], padding=self.config.conv2_padding) + conv2_b))
                # 池化
                pool2_b = tf.get_variable('pool2_b',
                    initializer=tf.constant(0.1, shape=[self.config.conv2_filter_num]))
                pool2_out = tf.nn.max_pool(conv2_out,
                    [1, self.config.conv2_pool_sizes[i], 1, 1], [1, self.config.conv2_pool_sizes[i], 1, 1],
                    padding=self.config.conv1_padding)
                pool2_out = tf.nn.tanh(pool2_out + pool2_b)

                outputs.append(pool2_out)

                # 加入正则项
                tf.add_to_collection('total_loss', 0.5 * self.config.l2_reg_lambda * tf.nn.l2_loss(conv1_W))
                tf.add_to_collection('total_loss', 0.5 * self.config.l2_reg_lambda * tf.nn.l2_loss(conv2_W))

        total_channels = len(self.config.conv2_filter_sizes) * self.config.conv2_filter_num

        if len(outputs) == 1:
            real_outputs = tf.reshape(outputs[0], [-1, total_channels * int(outputs[0].get_shape()[1])])
        else:
            raise ValueError('This version can only support one type of filter, '
                             'rather than {}'.format(len(self.config.conv2_filter_sizes)))

        # 加入FC层输出
        FC_W = tf.get_variable('FC_W', shape=[real_outputs.get_shape()[1], self.config.label_size],
                initializer=xavier_initializer())
        FC_b = tf.Variable(initial_value=tf.zeros([self.config.label_size]), name='FC_b')
        final_outputs = tf.matmul(real_outputs, FC_W) + FC_b
        tf.add_to_collection('total_loss', 0.5 * self.config.l2_reg_lambda * tf.nn.l2_loss(FC_W))

        return final_outputs


    # 损失节点
    def add_loss_op(self, outputs):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=outputs)
        loss = tf.reduce_mean(loss)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss


    def add_accu_op(self, outputs):
        accu = tf.argmax(tf.nn.softmax(outputs), axis=1)
        accu = tf.cast(tf.equal(accu, tf.argmax(self.labels, axis=1)), tf.float32)
        accu = tf.reduce_mean(accu)
        return accu


    # 训练节点
    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.learning_rate)
            train_op = opt.minimize(loss, self.global_step)
            return train_op


    def run_epoch(self, epoch_no, datapath):
        """

        run one epoch each time

        :param epoch_no: epoch number
        :param datapath: training data file path
        :param sample_num: the total number of samples from training data
        :return: None
        """
        print 'starting epoch {}'.format(epoch_no)
        total_batch = 0
        total_loss = 0.
        total_acc = 0.
        for i, (batch_x, batch_y) in enumerate(self.data_iterator(datapath)):
            _, loss, acc, summary = self.sess.run([self.train_op, self.loss,
                        self.accu, self.merged_summary_op],
                        feed_dict={self.inputs:batch_x, self.labels:batch_y})
            total_batch += 1
            self.summary_writer.add_summary(summary, (epoch_no - 1) * total_batch + i)
            total_loss += loss
            total_acc += acc
            if (i + 1) % self.config.batch_show == 0:
                print 'loss: {}, accu: {}'.format(total_loss / (i + 1), total_acc / (i + 1))

        print 'epoch {}, loss : {}, accu : {}'.format(epoch_no, total_loss / total_batch, total_acc / total_batch)


    def fit(self, datapath, test_datapath=None, restore=False):
        """

        fit models with training data set

        :param datapath: data file path
        :return: None
        """
        if restore:
            self.saver.restore(self.sess, self.model_path)
            print("Model restored from file: %s" % self.model_path)

        test_data = None
        test_labels = None
        if test_datapath is not None:
            test_data, test_labels = self.load_test_data(test_datapath)
        for i in xrange(self.config.epochs):
            self.run_epoch(i + 1, datapath)
            if (i + 1) % self.config.model_save_step == 0:
                save_path = self.saver.save(self.sess, self.model_path)
                print("Model saved in file: %s" % save_path)
            if test_data is not None and test_labels is not None:
                self.predict(test_data, test_labels)


    def predict(self, data, labels=None):
        if labels is not None:
            loss, acc, pred = self.sess.run([self.loss, self.accu, self.pred],
                        feed_dict={self.inputs:data, self.labels:labels})
            print '***test***\nloss : {}, accu : {}'.format(loss, acc)
            print confusion_matrix(np.argmax(labels, axis=1), pred)




if __name__ == '__main__':
    model = SignalModModel(Config(), '../log', '../model/model.pkg')
    model.fit(datapath='../data/5dB/train_mixSignals.txt', test_datapath='../data/5dB/test_mixSignals.txt')