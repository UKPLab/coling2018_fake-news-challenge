
"""
Implements a simple Feed Forward MLP that inherits the Sklearn BaseEstimator
"""
import numpy as np
import os.path as path
import os
import random

#https://github.com/fchollet/keras/issues/2280
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(12345)
random.seed(12345)

import shutil
import tensorflow as tf
from sklearn.base import BaseEstimator
from fnc.models.Keras_utils import calculate_class_weight

class MultiThreadingFeedForwardMLP(BaseEstimator):
    """
    Parameters
    ----------
    batch_size : int (default=200)
        Batch mode:             Number of samples = batch_size
        Mini-batch mode:        1 < batch_size < Number of samples
        Stochastic mode:        batch_size = 1
        Paper on this subject:  https://arxiv.org/abs/1609.04836
    """
    def __init__(self, pickle_file_ext=-1, save_folder=None, n_classes=4, batch_size=200, hm_epochs=15, keep_prob_const=1.0,
                 optimizer='adam', learning_rate=0.001, step_decay_LR=False, bias_init=0.0,
                 weight_init='xavier', hidden_layers=(600, 600, 600),
                 activation_function='relu', seed=12345, use_class_weights=False):
        self.pickle_file_ext = pickle_file_ext # underscore after var if it is computed (otherwise sklearn violation of estimator)
        self.n_classes = n_classes
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.hm_epochs = hm_epochs
        self.keep_prob_const = keep_prob_const
        self.optimizer = optimizer
        self.learning_rate_output = ""
        self.learning_rate = learning_rate
        self.step_decay_LR = step_decay_LR
        self.bias_init = bias_init
        self.weight_init = weight_init
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.seed = seed
        self.use_class_weights = use_class_weights

        # create config
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # if the model is called without a save folder and not by pickle.load(), just use a default
        # folder for temporary savings of the model
        if self.save_folder == None:
            self.save_folder = "%s/data/fnc-1/mlp_models/temp_models/" % (
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

        # search for a free folder
        tries = 1000
        for i in range(tries):
            try:
                self.random_file_ext_ = random.randint(1, 1000000)
                temp_feature_dir = self.save_folder + str(self.random_file_ext_) + "/"

                if not os.path.exists(temp_feature_dir):
                    self.feature_dir = temp_feature_dir
                    break
                elif i == tries-1:
                    raise ValueError('Please delete the folders in path=' + self.save_folder)
            except ValueError as err:
                print(err.args)

        if self.n_classes == 4:
            self.classes_ = np.asarray([0, 1, 2, 3])
        else:
            self.classes_ = np.asarray([0, 1])

        tf.reset_default_graph()


    def __reduce__(self):
        # Called by pickle.dump()
        return (MultiThreadingFeedForwardMLP, (self.random_file_ext_, self.save_folder, self.n_classes, self.batch_size, self.hm_epochs,
                                               self.keep_prob_const, self.optimizer, self.learning_rate, self.step_decay_LR, self.bias_init,
                                               self.weight_init, self.hidden_layers, self.activation_function, self.seed))

    def pelu(self, x):
        """
        Parametric Exponential Linear Unit (https://arxiv.org/abs/1605.09332v1).
        From: https://github.com/tensorflow/tensorflow/issues/7712
        """
        with tf.variable_scope(x.op.name + '_activation', initializer=tf.constant_initializer(1.0)):
            shape = x.get_shape().as_list()[1:]
            alpha = tf.get_variable('alpha', shape)
            beta = tf.get_variable('beta', shape)
            positive = tf.nn.relu(x) * alpha / (beta + 1e-9)
            negative = alpha * (tf.exp((-tf.nn.relu(-x)) / (beta + 1e-9)) - 1)
            return negative + positive

    def neural_network_model(self, X_train):

        # initialize variables
        self.x = tf.placeholder(tf.float32, shape=[None, len(X_train[0])])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.momentum = tf.placeholder(tf.float32)
        self.learning_rate_tensor = tf.placeholder(tf.float32)

        # create hidden layer list
        self.n_nodes_hl_list = [len(X_train[0])]
        for layer_size in self.hidden_layers:
            self.n_nodes_hl_list.append(layer_size)
        self.n_nodes_hl_list.append(self.n_classes)

        def get_layer(input_length, n_nodes_hl, hl_no, layer_input):
            """
            Returns a layer with the given parameters
            :param input_length: Size of the input of the layer
            :param n_nodes_hl: Number of nodes the layer should have
            :param hl_no: Index of the hidden layer, e.g. 1, 2, 3...
            :param weight_init: Method how the weights should be initialized;
            'xavier', 'sqrt_n' or leave empty for truncated normal with stddev of 0.1
            :param layer_input: The input for the layer; either a variable holding X or the previous layer
            :return: A layer with the given parameters
            """

            def weight_variable(shape, name):
                if self.weight_init == 'xavier':
                    return tf.get_variable(name, shape,
                                           initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))  # better initialization
                if self.weight_init == 'sqrt_n':

                    return tf.get_variable(name, shape,
                                           initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                      mode='FAN_IN',
                                                                                                      uniform=False,
                                                                                                      seed=self.seed))
                else:
                    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

            def bias_variable(shape, name):
                initial = tf.constant(self.bias_init, shape=shape)
                return tf.Variable(initial, name=name)

            # create weights and bias variables
            W = weight_variable([input_length, n_nodes_hl], "weight"+str(hl_no))
            self.weight_var_test = W
            b = bias_variable([n_nodes_hl], "bias"+str(hl_no))

            # if it's the last layer (output), do not add dropout and relu
            if i == len(self.n_nodes_hl_list)-2:
                layer = tf.add(tf.matmul(layer_input, W), b, name="prediction")
                return layer
            else:

                if self.activation_function == 'relu6':
                    layer = tf.nn.relu6(tf.add(tf.matmul(layer_input, W), b))
                elif self.activation_function == 'crelu':
                    layer = tf.nn.crelu(tf.add(tf.matmul(layer_input, W), b))
                elif self.activation_function == 'elu':
                    layer = tf.nn.elu(tf.add(tf.matmul(layer_input, W), b))
                elif self.activation_function == 'softplus':
                    layer = tf.nn.softplus(tf.add(tf.matmul(layer_input, W), b))
                elif self.activation_function == 'softsign':
                    layer = tf.nn.softsign(tf.add(tf.matmul(layer_input, W), b))
                elif self.activation_function == 'pelu':
                    layer = self.pelu(tf.add(tf.matmul(layer_input, W), b))
                else:
                    layer = tf.nn.relu(tf.add(tf.matmul(layer_input, W), b))

                dropout_layer = tf.nn.dropout(layer, self.keep_prob)
                return dropout_layer

        # create all layers; input is the input for the next layer
        layer_input = self.x
        for i in range(len(self.n_nodes_hl_list)-1):
            layer_input = get_layer(self.n_nodes_hl_list[i], self.n_nodes_hl_list[i+1], i, layer_input)

        prob = tf.nn.softmax(layer_input)
        return layer_input, prob

    def fit(self, X_train, y_train):

        def save_graph(self, sess):

            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

            if not os.path.exists(self.save_folder + str(self.random_file_ext_) + "/"):
                os.makedirs(self.save_folder + str(self.random_file_ext_) + "/")
            permanent_saver = tf.train.Saver()
            permanent_saver.save(sess, self.save_folder + str(self.random_file_ext_) + "/" + "model")

        def convert_data_to_one_hot(y_train):
            #y_test_temp = np.zeros((y_test.size, y_test.max() + 1), dtype=np.int)
            #y_test_temp[np.arange(y_test.size), y_test] = 1

            # Other option:
            #   y_train is a tensor then because of one_hot, but feed_dict only accepts numpy arrays => replace y_train with sess.run(y_train)
            #   http://stackoverflow.com/questions/34410654/tensorflow-valueerror-setting-an-array-element-with-a-sequence
            # return tf.one_hot(y_train, 4), tf.one_hot(y_test, 4)
            y_train_temp = np.zeros((y_train.size, y_train.max() + 1), dtype=np.int)
            y_train_temp[np.arange(y_train.size), y_train] = 1

            return y_train_temp

        y_train_conv = convert_data_to_one_hot(y_train)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.prediction, self.prob = self.neural_network_model(
                X_train)


            if self.use_class_weights == True:#https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow#answer-38912982
                class_weights = calculate_class_weight(y_train) # doesnt work really well
                class_weight_mod = tf.constant(
                    [
                        [class_weights[0], class_weights[1]]
                    ])
                weight_per_label = tf.transpose(tf.matmul(self.y, tf.transpose(class_weight_mod)))
                xent = tf.multiply(weight_per_label
                                   , tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
            else:
                xent = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y)

            cost = tf.reduce_mean(xent)

            # simple explanations to some optimizers:
            #   http://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
            #   http://cs231n.github.io/neural-networks-3/
            #   http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
            # Parameters: http://tflearn.org/optimizers
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tensor).minimize(cost)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate_tensor).minimize(cost)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate_tensor).minimize(cost)
            elif self.optimizer == 'graddesc':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_tensor).minimize(cost)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_tensor,
                                                       momentum=self.momentum).minimize(cost)
            elif self.optimizer == 'nesterov_momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_tensor, momentum=self.momentum,
                                                       use_nesterov=True).minimize(cost)
            elif self.optimizer == 'proxada':
                optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.learning_rate_tensor).minimize(cost)
            elif self.optimizer == 'rms':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_tensor).minimize(cost)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tensor).minimize(cost)

        with tf.Session(graph=self.graph, config=self.config) as sess:
            sess.run(tf.global_variables_initializer())

            momentum_start = 0.5
            momentum_end = 0.99
            calc_learning_rate = self.learning_rate

            for epoch in range(self.hm_epochs):
                epoch_loss = 0

                # increase momentum steadily
                i = 0
                calc_momentum = momentum_start + (
                    float((momentum_end - momentum_start) / self.hm_epochs) * epoch)  # increase momentum with epochs

                if self.step_decay_LR == True and (epoch == 20 or epoch == 35 or epoch == 45) and epoch > 0:
                    calc_learning_rate = float(calc_learning_rate / 10.0)

                while i < len(X_train):
                    start = i
                    end = i + self.batch_size
                    batch_x = np.array(X_train[start:end])
                    batch_y = np.array(y_train_conv[start:end])

                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x,
                                                                  self.y: batch_y,
                                                                  self.keep_prob: self.keep_prob_const,
                                                                  # self.learning_rate: CHANGE EVERY FEW EPOCHS,
                                                                  self.momentum: calc_momentum,
                                                                  self.learning_rate_tensor: calc_learning_rate
                                                                  })
                    epoch_loss += c
                    i += self.batch_size

                self.learning_rate_output += str(epoch_loss) + "\n"
                print('Epoch', epoch + 1, 'completed out of', self.hm_epochs, 'loss:', epoch_loss, 'LR=',
                      calc_learning_rate)
                # get second weight matrix
                # test = sess.run(self.graph.get_tensor_by_name("weight1:0"))
                # print(test)

            # save the graph permanently
            save_graph(self, sess)

            # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # print('Accuracy by TensorFlow:', accuracy.eval({x: X_test, y: y_test, keep_prob: keep_prob_const}))
            # value = sess.run(tf.argmax(self.prediction, 1), feed_dict={self.x: X_test, self.keep_prob: self.keep_prob_const})
        # return value
        return self



    def predict(self, X_test):

        def predict_from_restored(X_test, file_path):
            """
            Predicts y for given X_test. Folder in which the stored model was saved has to be
            provided.
            :param X_test: The test features
            :param folder: The folder in which the model has been saved
            :return: The prediction y for X_test
            """

            # handle absolute file_path
            folder_name = file_path.split("mlp_models/")[1]  # returns model folder name
            #folder_name.replace("voting_mlps_hard_final_2", "voting_mlps_hard_final_BEST") # hard coded fix for final model
            mlp_folder_path = "%s/data/fnc-1/mlp_models/" % (
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            file_path = mlp_folder_path + folder_name

            # https://kevincodeidea.wordpress.com/2016/08/02/tensorflow-save-and-load-a-model-in-a-serious-way-from-different-files/
            # Also check:
            #   http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
            #   https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_class2/tf_with_save.py
            with tf.Graph().as_default() as g:
                with tf.Session(config=self.config) as sess:
                    prediction, prob = self.neural_network_model(X_test)
                    saver = tf.train.Saver()
                    saver.restore(sess, file_path)
                    value = sess.run(tf.argmax(prediction, 1),
                                     feed_dict={self.x: X_test, self.keep_prob: self.keep_prob_const})
            return value

        if self.pickle_file_ext != -1:
            return predict_from_restored(X_test, self.save_folder + str(self.pickle_file_ext) + "/" + "model")
        else:
            with tf.Session(graph=self.graph, config=self.config) as sess:
                new_saver = tf.train.import_meta_graph(self.feature_dir + "model.meta")
                new_saver.restore(sess, self.feature_dir + "model")
                value = sess.run(tf.argmax(self.prediction, 1), feed_dict={self.x: X_test, self.keep_prob: self.keep_prob_const})
                #value2 = sess.run(self.prediction, feed_dict={self.x: X_test, self.keep_prob: self.keep_prob_const})
                #print(value2[0])
            return value



    def predict_proba(self, X):
        """with tf.Graph().as_default() as g:
            with tf.Session() as sess:
                prediction, prob = self.neural_network_model(X)
                sess.run(tf.initialize_all_variables())
                saver = tf.train.Saver()
                value = sess.run(tf.argmax(prediction, 1),
                                 feed_dict={self.x: X, self.keep_prob: self.keep_prob_const})
                print(sess.run(g.get_tensor_by_name("weight1:0")))"""

        def predict_proba_from_restored(X_test, file_path):
            """
            Predicts y for given X_test. Folder in which the stored model was saved has to be
            provided.
            :param X_test: The test features
            :param folder: The folder in which the model has been saved
            :return: The prediction y for X_test
            """

            # handle absolute file_path
            folder_name = file_path.split("mlp_models/")[1]  # returns model folder name
            #folder_name.replace("voting_mlps_hard_final_2", "voting_mlps_hard_final_BEST") # hard coded fix for final model
            mlp_folder_path = "%s/data/fnc-1/mlp_models/" % (
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            file_path = mlp_folder_path + folder_name

            with tf.Graph().as_default() as g:
                with tf.Session(config=self.config) as sess:
                    prediction, prob = self.neural_network_model(X_test)
                    saver = tf.train.Saver()
                    saver.restore(sess, file_path)
                    value = sess.run(prob, feed_dict={self.x: X_test, self.keep_prob: self.keep_prob_const})
            return value

        # if the pickle_file_extension is not -1, the model was called by pickle and hence should be restored from disk
        if self.pickle_file_ext != -1:
            return predict_proba_from_restored(X, self.save_folder + str(self.pickle_file_ext) + "/" + "model")
        else:
            with tf.Session(graph=self.graph, config=self.config) as sess:
                new_saver = tf.train.import_meta_graph(self.feature_dir + "model.meta")
                new_saver.restore(sess, self.feature_dir + "model")
                value = sess.run(self.prob, feed_dict={self.x: X, self.keep_prob: self.keep_prob_const})
            return value

    def delete_saved_graph(self):
        if (os.path.exists(self.feature_dir)):
            try:
                shutil.rmtree(self.feature_dir)
            except Exception as e:
                print(e)

    def get_learning_rates(self, fold):
        return str(fold) + "\n" + self.learning_rate_output


"""
Saving model:
    https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops/saving_and_restoring_variables
    https://www.tensorflow.org/versions/r0.12/how_tos/variable_scope/
    https://www.tensorflow.org/programmers_guide/variables
    https://www.tensorflow.org/programmers_guide/meta_graph
    https://nathanbrixius.wordpress.com/2016/05/24/checkpointing-and-reusing-tensorflow-models/
    http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables
    stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
        https://www.tensorflow.org/api_docs/python/tf/train/latest_checkpoint
"""
