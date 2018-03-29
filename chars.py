import re
import math
import nltk
import codecs
import pickle
import os, sys
import numpy as np
import datetime as dt
import tensorflow as tf
import collections as coll

# As corpus contains shakeshpeer, our life is truely hard
reload(sys)
sys.setdefaultencoding('utf-8')

SOS0 = '<sos>'
EOS0 = '<eos>'

unknown = '<unk>'
freq_threshold = 1

data_path = '/home1/e1-246-54/lstm/NLU-LSTM/char_level/'

def get_vocab(fname = 'gutenberg.train', perc = 1.0):
    # corpus_file = fname

    with codecs.open(fname, "r", encoding="latin-1", errors="ignore") as f:
        text = list(f.read())
        text = np.array(text)[:int(len(text) * perc)]

    # cfp = open(corpus_file, 'r')
    # text = re.findall(r"[\w'<>]+|[.,!?;]", cfp.read())
    # cfp.close()

    # Get unigram counts. This also counts SOS, EOS which is not informative
    unicnts = coll.Counter(text)

    unknown_map = []
    total = 0
    for entry in unicnts.items():
        if entry[1] < freq_threshold:
            unknown_map.append(entry[0])
            total += entry[1]

    for entry in unknown_map:
        unicnts.pop(entry)

    itemset = unicnts.items()
    itemset.append((unknown, total))
    words = sorted(itemset, key=lambda x: (-x[1], x[0]))
    words,_ = list(zip(*words))
    word_map = dict(zip(words, range(len(words))))
    word_unmap = dict(zip(word_map.values(), word_map.keys()))

    return word_map, word_unmap

def encode_data(fname, word_map, perc = 1.0):
    # fp = open(fname, 'r')

    with codecs.open(fname, "r", encoding="latin-1", errors="ignore") as f:
        text = list(f.read())
        text = np.array(text)[:int(len(text) * perc)]

    # text = re.findall(r"[\w'<>]+|[.,!?;]", fp.read())

    code = []
    for word in text:
        if word in word_map:
            code.append(word_map[word])
        else:
            code.append(word_map[unknown])
    return np.array(code)

# Generate batches of input from text which are fed to LSTM during training.
# This involves spliting data into rows of batch_size, and again split each row
# into step_len words. The next word is considered as target.
def generate_input(train_data, batch_size, step_len):
    train_data = tf.convert_to_tensor(train_data,\
                                        name = "train_data", dtype = tf.int32)
    data_size = tf.size(train_data)
    batch_len = data_size // batch_size
    data = tf.reshape(train_data[0: batch_size * batch_len],\
                                            [batch_size, batch_len])

    num_strides = (batch_len - 1) // step_len

    ii = tf.train.range_input_producer(num_strides, shuffle = False).dequeue()
    X = data[:, ii * step_len: (ii + 1) * step_len]
    X.set_shape([batch_size, step_len])
    # Y = data[:, ii * step_len + 1: (ii + 1) * step_len + 1]
    Y = data[:, (ii + 1) * step_len + 1]
    # Y.set_shape([batch_size, step_len])
    return X, Y

class Input(object):
    def __init__(self, train_data, batch_size, step_len):
        self.batch_size = batch_size
        self.step_len = step_len
        self.batch_len = len(train_data) // step_len
        self.num_strides = (self.batch_len - 1) // step_len
        self.data, self.targets = generate_input(train_data,
                                                        batch_size, step_len)

class Model(object):
    def __init__(self, datain, is_training,
                hidden_size, vocabulary,
                num_layers = 1, init_scale = 0.5, dropout = 0.6,
                step_len = 32, mode = 'test'):
        self.datain = datain
        if is_training == True or is_training == False and mode == 'test':
            self.data, self.targets = self.datain.data, self.datain.targets
            self.num_strides = datain.num_strides
        else:
            if mode == 'generate':
                print datain.batch_size, step_len
                self.data = tf.placeholder(tf.int32,
                                        [datain.batch_size, step_len])
        # self.data, self.targets = self.datain.data, self.datain.targets
        self.step_len = datain.step_len
        self.batch_size = datain.batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        embedding = tf.Variable(tf.random_uniform(
                        [vocabulary, hidden_size], -init_scale, init_scale),
                        name = "embedding")
        inputs = tf.nn.embedding_lookup(embedding, self.data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        self.init_state = tf.placeholder(tf.float32,
                            [num_layers, 2, self.batch_size, self.hidden_size])
        state_per_layer_list = tf.unstack(self.init_state, axis = 0)
        rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(
                                            state_per_layer_list[idx][0],
                                            state_per_layer_list[idx][1])\
                                            for idx in range(self.num_layers)])

        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias = 1.0)
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                output_keep_prob = dropout)
        cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)],
                                                        state_is_tuple = True)
        output, self.state = tf.nn.dynamic_rnn(cell,
                    inputs, dtype=tf.float32, initial_state = rnn_tuple_state)
        out1 = output[:, self.step_len - 1,:]

        # format output for softmax: we are only taking last layer output
        output = tf.reshape(out1, [-1, hidden_size])
        # output = tf.reshape(output, [-1, hidden_size])
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocabulary],
                                                    -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocabulary],
                                                    -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # logits = tf.reshape(logits,
        #                         [self.batch_size, self.step_len, vocabulary])
        # loss = tf.contrib.seq2seq.sequence_loss(
        #         logits,
        #         self.targets,
        #         tf.ones([self.batch_size, self.step_len], dtype=tf.float32),
        #         average_across_timesteps = False,
        #         average_across_batch = True)

        # Calculate loss
        logits = tf.reshape(logits, [self.batch_size, vocabulary])
        if mode != 'generate':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        labels = self.targets,
                                                        logits = logits,
                                                        name = 'loss_function')
            self.cost = tf.reduce_sum(loss)

        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocabulary]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        if mode != 'generate':
            correct_prediction = tf.equal(self.predict,
                                        tf.reshape(self.targets, [-1]))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
           return

        self.learning_rate = tf.Variable(0.0, trainable = False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

def train(train_data, vocabulary, num_layers,
            num_epochs, batch_size, model_save_name,
            step_len = 32, hidden_size = 512,
            learning_rate = 1.0, max_lr_epoch = 10,
            lr_decay = 0.93, print_iter = 50):
    # setup data and models
    training_input = Input(batch_size = batch_size,
                                step_len = step_len, train_data = train_data)
    m = Model(training_input, is_training = True, hidden_size = hidden_size,
                vocabulary = vocabulary, num_layers = num_layers)

    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        saver = tf.train.Saver()

        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            current_state =\
                        np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            for step in range(training_input.num_strides):
                if step % print_iter != 0:
                    cost, _, current_state =\
                                    sess.run([m.cost, m.train_op, m.state],
                                    feed_dict = {m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds)\
                                                                / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _, current_state, acc = sess.run(\
                                    [m.cost, m.train_op, m.state, m.accuracy],
                                    feed_dict = {m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch, step, cost, acc, seconds))

            # save a model checkpoint
            saver.save(sess, data_path + model_save_name, global_step=epoch)
        # do a final save
        saver.save(sess, data_path + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)

def test(test_data, vocabulary, batch_size, model_path,\
            step_len = 32, hidden_size = 512, num_layers = 1):
    # setup data and models
    test_input = Input(batch_size = batch_size,
                                step_len = step_len, train_data = test_data)
    m = Model(test_input, is_training = False, hidden_size = hidden_size,
                vocabulary = vocabulary, num_layers=num_layers, mode = "test")

    perplexity = 0

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        # restore the trained model
        saver.restore(sess, model_path)
        print m.cost
        m.batch_size = batch_size
        m.hidden_size = hidden_size
        current_state = np.zeros((num_layers, 2, m.batch_size, m.hidden_size))

        cnt = 0
        print m.num_strides
        for batch in range(m.num_strides - 1):
            # if batch % step_len == 0:
            #     current_state = np.zeros((num_layers, 2, m.batch_size, m.hidden_size))

            softmax_out, current_state, targets, accuracy =\
                            sess.run([m.softmax_out, m.state, m.targets, m.accuracy],
                            feed_dict = {m.init_state: current_state})
            prob = softmax_out[0][targets[0]]
            perplexity += np.log(prob)
            mlw = np.argmax(softmax_out)
            print batch, accuracy, prob, np.log(prob)

            cnt += 1
#            if batch == 2 * step_len:
#                break

        print 'Test perplexity =', -perplexity / m.num_strides, -perplexity / cnt
        # np.exp((-1.0 / m.num_strides) * perplexity)
        # close threads
        coord.request_stop()
        coord.join(threads)

def generate_text(word_map, word_unmap, vocabulary, batch_size, model_path,\
            step_len = 32, hidden_size = 512, num_layers = 1):

#    seed = encode_data('seed_file', word_map)
    seed = np.array([word_map['i']]).reshape((1, 1))
    seed = seed.reshape((1, len(seed)))
    tmp = seed.shape[1]
    # setup data and models
    test_input = Input(batch_size = batch_size,
                                step_len = len(seed), train_data = seed)
    m = Model(test_input, is_training = False, hidden_size = hidden_size, step_len = tmp,
            vocabulary = vocabulary, num_layers=num_layers, mode = "generate")

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        # restore the trained model
        saver.restore(sess, model_path)
        m.batch_size = batch_size
        m.hidden_size = hidden_size
        current_state = np.zeros((num_layers, 2, m.batch_size, m.hidden_size))
        softmax_out, current_state =\
                        sess.run([m.softmax_out, m.state],
                        feed_dict = {m.data: seed, m.init_state: current_state})
        ptr = ''
        for ii in range(30):
            # print ii
            mlw = np.argmax(softmax_out)
            # print softmax_out, mlw
            ptr += word_unmap[mlw]
            seed = seed[0][1:].tolist()
            seed.append(mlw)
            seed = np.array(seed).reshape((1, tmp))

            softmax_out, current_state =\
                            sess.run([m.softmax_out, m.state],
                            feed_dict = {m.data: seed, m.init_state: current_state})

        # close threads
        print ptr
        coord.request_stop()
        coord.join(threads)

step_len = 8
num_layers = 1
num_epochs = 15
batch_size = 1000
init_scale = 0.05
hidden_size = 700
#train_mode = True
train_mode = False
model_save_name = 'low_data'

word_map, word_unmap = get_vocab('/home1/e1-246-54/lstm/NLU-LSTM/gutenberg.train')
vocabulary = len(word_map)
train_data = encode_data('/home1/e1-246-54/lstm/NLU-LSTM/gutenberg.train', word_map)
test_data = encode_data('/home1/e1-246-54/lstm/NLU-LSTM/gutenberg.test', word_map)

if train_mode == True:
    print 'Training Model'
    train(train_data = train_data, vocabulary = vocabulary,\
                            num_layers = num_layers,\
                            num_epochs = num_epochs, batch_size = batch_size,
                            model_save_name = model_save_name,
                            hidden_size = hidden_size)
#    test(test_data, vocabulary, batch_size = 1,\
#        model_path = data_path + model_save_name + '-final',
#        step_len = step_len, hidden_size = hidden_size,\
#        num_layers = num_layers)

else:
    #test(test_data, vocabulary, batch_size = 8,\
    #   model_path = data_path + model_save_name + '-9',
    #   step_len = step_len, hidden_size = hidden_size,\
    #   num_layers = num_layers)
    generate_text(word_map, word_unmap, vocabulary, batch_size = 1,\
        model_path = data_path + model_save_name + '-8',
        step_len = 8, hidden_size = hidden_size,\
        num_layers = num_layers)
