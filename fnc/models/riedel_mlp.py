from sklearn.base import BaseEstimator
import random
import tensorflow as tf
import os

## Based on model from https://github.com/uclmr/fakenewschallenge/blob/master/
## merged with model from Benjamin Schiller Tu Darmstadt
class riedel_mlp(BaseEstimator):
    def __init__(self, pickle_file_ext=-1, save_folder=None, r=random.Random(), lim_unigram=5000,
                 target_size=4, hidden_size=100, train_keep_prob=0.6, l2_alpha=0.00001, learn_rate=0.01,
                 clip_ratio=5, batch_size_train=500, epochs=90):
        self.pickle_file_ext = pickle_file_ext
        self.save_folder = save_folder
        self.r = r
        self.lim_unigram = lim_unigram
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.train_keep_prob = train_keep_prob
        self.l2_alpha = l2_alpha
        self.learn_rate = learn_rate
        self.clip_ratio = clip_ratio
        self.batch_size_train = batch_size_train
        self.epochs = epochs

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        #self.config.gpu_options.per_process_gpu_memory_fraction = 0.2

        # taken from Benjamins implementation
        # if the model is called without a save folder and not by pickle.load(), just use a default
        # folder for temporary savings of the model

        tf.reset_default_graph()

    def __reduce__(self):
        #Neccessary to be able to pickle the model
        return (riedel_mlp, (self.pickle_file_ext, self.save_folder, self.r, self.lim_unigram, self.target_size,
                             self.hidden_size, self.train_keep_prob, self.l2_alpha, self.learn_rate, self.clip_ratio,
                             self.batch_size_train, self.epochs))

    # Define model
    def initialize_neural_network_model(self, X_train):
        # Create placeholders
        self.features_pl = tf.placeholder(tf.float32, [None, len(X_train[0])], 'features')
        self.stances_pl = tf.placeholder(tf.int64, [None], 'stances')
        self.keep_prob_pl = tf.placeholder(tf.float32)

        # Infer batch size
        self.batch_size = tf.shape(self.features_pl)[0]

        # Define multi-layer perceptron
        self.hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(self.features_pl, self.hidden_size)), keep_prob=self.keep_prob_pl)
        self.logits_flat = tf.nn.dropout(tf.contrib.layers.linear(self.hidden_layer, self.target_size), keep_prob=self.keep_prob_pl)
        self.logits = tf.reshape(self.logits_flat, [self.batch_size, self.target_size])

        # Define L2 loss
        self.tf_vars = tf.trainable_variables()
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.tf_vars if 'bias' not in v.name]) * self.l2_alpha

        # Define overall loss
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.stances_pl) + self.l2_loss)

        # Define prediction
        self.softmaxed_logits = tf.nn.softmax(self.logits)
        self.predict_value = tf.arg_max(self.softmaxed_logits, 1)

    def fit(self, X_train, y_train, sample_weight=None):
        def save_session(self, sess):
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

            permanent_saver = tf.train.Saver()
            permanent_saver.save(sess, self.save_folder+"model")
        '''
        def convert_data_to_one_hot(y_train):
            y_train_temp = np.zeros((y_train.size, y_train.max() + 1), dtype=np.int)
            y_train_temp[np.arange(y_train.size), y_train] = 1
            return y_train_temp

        y_train_conv = convert_data_to_one_hot(y_train)
        '''

        self.graph = tf.Graph()

        #initilize model
        with self.graph.as_default():
            self.initialize_neural_network_model(X_train)

            # Define optimiser
            opt_func = tf.train.AdamOptimizer(self.learn_rate)
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tf_vars), self.clip_ratio)
            opt_op = opt_func.apply_gradients(zip(grads, self.tf_vars))

        # Perform training
        with tf.Session(graph=self.graph, config=self.config) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epochs):
                total_loss = 0
                indices = list(range(len(X_train)))
                self.r.shuffle(indices)

                for i in range(len(X_train) // self.batch_size_train):
                    batch_indices = indices[i * self.batch_size_train: (i + 1) * self.batch_size_train]
                    batch_features = [X_train[i] for i in batch_indices]
                    batch_stances = [y_train[i] for i in batch_indices]

                    batch_feed_dict = {self.features_pl: batch_features, self.stances_pl: batch_stances, self.keep_prob_pl: self.train_keep_prob}
                    _, current_loss = sess.run([opt_op, self.loss], feed_dict=batch_feed_dict)
                    total_loss += current_loss
            save_session(self, sess)
        return self

    def predict(self, X_test):

        def load_model(sess):
            new_saver = tf.train.import_meta_graph(self.save_folder + "model.meta")
            new_saver.restore(sess, self.save_folder + "model")

        with tf.Graph().as_default() as g:
            with tf.Session(config=self.config) as sess:
                self.initialize_neural_network_model(X_test)
                saver = tf.train.Saver()
                saver.restore(sess, self.save_folder + "model")
                test_feed_dict = {self.features_pl: X_test, self.keep_prob_pl: 1.0}
                value = sess.run(self.predict_value, feed_dict=test_feed_dict)
        return value



