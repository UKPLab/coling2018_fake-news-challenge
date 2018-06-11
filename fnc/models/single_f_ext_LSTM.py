import numpy as np
import os.path as path
import pickle

from keras.layers.core import Dense
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras import optimizers
from fnc.models.Keras_utils import EarlyStoppingOnF1, convert_data_to_one_hot, calculate_class_weight, split_X
from fnc.models.keras_custom_layers.attention_custom import *
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras.layers import Embedding, Input
import tensorflow as tf

class single_f_ext_LSTM():
    """
    Rebiuld of Basic LSTM from https://web.stanford.edu/class/cs224n/reports/2748568.pdf
    """
    def __init__(self, epochs=70, batch_size=200, param_dict="flat_LSTM_100d_50_500", lr=0.001, optimizer="adam", lr_decay=0.0,
                 dropout_LSTM=0.2, LSTM_kernel_regularizer=None, LSTM_kernel_constraint=None, recurrent_dropout_LSTM=0.0, LSTM_activity_regularizer=None,
                 dense_activity_regularizer=None, min_epoch=10, gpu_memory_fraction=0.3, seed=12345, use_class_weights=True, MAX_SEQ_LENGTH=None, save_folder=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.config = tf.ConfigProto()
        #self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.graph = None
        self.sess = None
        self.regularizer = None
        self.regularizer_emb = None #l2(0.01)
        self.kernel_initializer = 'glorot_uniform'
        self.lr = lr
        self.optimizer_name = optimizer
        self.lr_decay = lr_decay
        self.LSTM_implementation = 2 # faster but slightly reduced regularization
        self.LSTM_return_sequences = False
        self.mask_zero = True
        self.min_epoch = min_epoch
        self.seed = seed
        self.use_class_weights = use_class_weights
        self.save_folder = save_folder
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH

        # LSTM regularization
        self.LSTM_kernel_regularizer = LSTM_kernel_regularizer
        self.LSTM_kernel_constraint = LSTM_kernel_constraint
        self.LSTM_activity_regularizer = LSTM_activity_regularizer
        self.dropout_LSTM = dropout_LSTM
        self.recurrent_dropout_LSTM = recurrent_dropout_LSTM
        self.dense_activity_regularizer = dense_activity_regularizer
        self.trainable_emb = True

        # location path for features
        self.FEATURES_DIR = "%s/data/fnc-1/features/" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

        # param_dict file
        self.PARAM_DICT_FILENAME = param_dict + "_param_dict.pkl"
        print("PARAM_DICT LOADED="+self.PARAM_DICT_FILENAME)

        if self.save_folder is None:
            self.save_folder = "%s/data/fnc-1/mlp_models/temp_models/" % (
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    def __reduce__(self):
        # Called by pickle.dump()
        return (self.__class__, (self.epochs, self.batch_size, "flat_LSTM_100d_50_500", self.lr, "adam", self.lr_decay,
                                self.dropout_LSTM, self.LSTM_kernel_regularizer, self.LSTM_kernel_constraint,
                                self.recurrent_dropout_LSTM, self.LSTM_activity_regularizer, self.dense_activity_regularizer,
                                self.min_epoch, 0.3, self.seed, self.use_class_weights, self.MAX_SEQ_LENGTH, self.save_folder))


    def step_decay(self, epoch):
        """
        Drop LR each epochs_drop by drop_rate.
        :param epoch:
        :return:
        """
        lr = K.get_value(self.model.optimizer.lr)
        new_lr = lr
        if epoch > 0 and float(epoch) % 10 == 0:
            new_lr = float(lr) / 10.0
        #print("\n")
        print("Current LR " + str(lr) + " => New LR " + str(new_lr))
        #print("\n")
        return new_lr


    def fit(self, X_train, y_train, X_test, y_test, test_fold, loss_filename):
        self.test_fold = test_fold
        self.y_test = y_test

        # set session config with gpu variable growth
        self.sess = tf.Session(config=self.config) # see https://github.com/fchollet/keras/issues/1538
        K.set_session(self.sess)

        # convert y_train to one-hot vector
        y_train_one_hot = convert_data_to_one_hot(y_train)
        y_test_one_hot = convert_data_to_one_hot(self.y_test)

        # load feature dict for LSTM_1000_GloVe
        with open(self.FEATURES_DIR+self.PARAM_DICT_FILENAME, "rb") as f:
            param_dict = pickle.load(f)

        # load parameters needed for embedding layer
        EMBEDDING_DIM = param_dict["EMBEDDING_DIM"] # e.g. 50
        self.MAX_SEQ_LENGTH = param_dict["MAX_SEQ_LENGTH"] # e.g. 100

        X_train_LSTM, X_train_MLP = split_X(X_train, self.MAX_SEQ_LENGTH)
        X_test_LSTM, X_test_MLP = split_X(X_test, self.MAX_SEQ_LENGTH)

        # load embeddings
        EMBEDDING_FILE = np.load(self.FEATURES_DIR+param_dict["EMBEDDING_FILE"])

        print("EMBEDDING_FILE.shape = " + str(EMBEDDING_FILE.shape))


        # calc cass weights
        class_weights = calculate_class_weight(y_train, no_classes=4)

        ################
        # CLAIMS LAYER #
        ################
        lstm_input = Input(shape=(self.MAX_SEQ_LENGTH,), dtype='int32', name='lstm_input') # receive sequences of MAX_SEQ_LENGTH_CLAIMS integers
        embedding = Embedding(input_dim=len(EMBEDDING_FILE), # lookup table size
                                    output_dim=EMBEDDING_DIM, # output dim for each number in a sequence
                                    weights=[EMBEDDING_FILE],
                                    input_length=self.MAX_SEQ_LENGTH, # receive sequences of MAX_SEQ_LENGTH_CLAIMS integers
                                    mask_zero=True,
                                    trainable=True)(lstm_input)

        data_LSTM = LSTM(
            100, return_sequences=True, stateful=False, dropout=0.2,
            batch_input_shape=(self.batch_size, self.MAX_SEQ_LENGTH, EMBEDDING_DIM),
            input_shape=(self.MAX_SEQ_LENGTH, EMBEDDING_DIM), implementation=self.LSTM_implementation
            )(embedding)
        data_LSTM = LSTM(
            100, return_sequences=False, stateful=False, dropout=0.2,
            batch_input_shape=(self.batch_size, self.MAX_SEQ_LENGTH, EMBEDDING_DIM),
            input_shape=(self.MAX_SEQ_LENGTH, EMBEDDING_DIM), implementation=self.LSTM_implementation
            )(data_LSTM)

        ###############################
        # MLP (NON-TIMESTEP) FEATURES #
        ###############################
        mlp_input = Input(shape=(len(X_train_MLP[0]),), dtype='float32', name='mlp_input')

        ###############
        # MERGE LAYER #
        ###############
        merged = concatenate([data_LSTM, mlp_input])

        dense_mid = Dense(600, kernel_regularizer=self.regularizer, kernel_initializer=self.kernel_initializer,
                          activity_regularizer=self.dense_activity_regularizer, activation='relu')(merged)
        dense_mid = Dense(600, kernel_regularizer=self.regularizer, kernel_initializer=self.kernel_initializer,
                          activity_regularizer=self.dense_activity_regularizer, activation='relu')(dense_mid)
        dense_mid = Dense(600, kernel_regularizer=self.regularizer, kernel_initializer=self.kernel_initializer,
                          activity_regularizer=self.dense_activity_regularizer, activation='relu')(dense_mid)
        dense_out = Dense(4,activation='softmax', name='dense_out')(dense_mid)

        # build model
        self.model = Model(inputs=[lstm_input, mlp_input], outputs=[dense_out])

        # print summary
        self.model.summary()

        # optimizers
        if self.optimizer_name == "adagrad":
            optimizer = optimizers.Adagrad(lr=self.lr)
            print("Used optimizer: adagrad, lr="+str(self.lr))
        elif self.optimizer_name == "adamax":
            optimizer = optimizers.Adamax(lr=self.lr)
            print("Used optimizer: adamax, lr="+str(self.lr))
        elif self.optimizer_name == "nadam":
            optimizer = optimizers.Nadam(lr=self.lr)  # recommended to leave at default params
            print("Used optimizer: nadam, lr="+str(self.lr))
        elif self.optimizer_name == "rms":
            optimizer = optimizers.RMSprop(lr=self.lr)  # recommended for RNNs
            print("Used optimizer: rms, lr="+str(self.lr))
        elif self.optimizer_name == "SGD":
            optimizer = optimizers.SGD(lr=self.lr)  # recommended for RNNs
            print("Used optimizer: SGD, lr="+str(self.lr))
        elif self.optimizer_name == "adadelta":
            optimizer = optimizers.Adadelta(self.lr)  # recommended for RNNs
            print("Used optimizer: adadelta, lr="+str(self.lr))
        else:
            optimizer = optimizers.Adam(lr=self.lr)
            print("Used optimizer: Adam, lr=" + str(self.lr))

        # compile model
        self.model.compile(optimizer, 'kullback_leibler_divergence', # categorial_crossentropy
                           metrics=['accuracy']
                           )
        if self.use_class_weights == True:
            self.model.fit([X_train_LSTM, X_train_MLP],
                             y_train_one_hot,
                             validation_data=([X_test_LSTM, X_test_MLP], y_test_one_hot),
                             batch_size=self.batch_size, epochs=self.epochs, verbose=1, class_weight=class_weights,
                           callbacks=[
                               EarlyStoppingOnF1(self.epochs,
                                                 X_test_LSTM, X_test_MLP, self.y_test,
                                                 loss_filename, epsilon=0.0, min_epoch=self.min_epoch),
                           ]
                           )
        else:
            self.model.fit([X_train_LSTM, X_train_MLP],
                             y_train_one_hot,
                             validation_data=([X_test_LSTM, X_test_MLP], y_test_one_hot),
                             batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                           callbacks=[
                               EarlyStoppingOnF1(self.epochs,
                                                 X_test_LSTM, X_test_MLP, self.y_test,
                                                 loss_filename, epsilon=0.0, min_epoch=self.min_epoch),
                           ]
                           )
        self.model.save(self.save_folder+"save.h5")
        return self

    def predict(self, X_test):
        print("Loading model from:" + self.save_folder+"save.h5")
        self.model = load_model(self.save_folder+"save.h5")
        if (self.model != None):
            X_test_LSTM, X_test_MLP = split_X(X_test, self.MAX_SEQ_LENGTH)

            predicted_one_hot = self.model.predict([X_test_LSTM, X_test_MLP])
            predicted_int = np.argmax(predicted_one_hot, axis=-1)

            return predicted_int
        else:
            print("No trained model found.")
            return np.zeros(len(X_test))

class single_f_ext_LSTM_att():
    """
    Rebiuld of Basic LSTM from https://web.stanford.edu/class/cs224n/reports/2748568.pdf
    """
    def __init__(self, epochs=70, batch_size=200, param_dict="flat_LSTM_100d_50_500", lr=0.001, optimizer="adam", lr_decay=0.0,
                 dropout_LSTM=0.2, LSTM_kernel_regularizer=None, LSTM_kernel_constraint=None, recurrent_dropout_LSTM=0.0, LSTM_activity_regularizer=None,
                 dense_activity_regularizer=None, min_epoch=10, gpu_memory_fraction=0.3, seed=12345, use_class_weights=True, MAX_SEQ_LENGTH=None, save_folder=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.graph = None
        self.sess = None
        self.regularizer = None
        self.regularizer_emb = None
        self.kernel_initializer = 'glorot_uniform'
        self.lr = lr
        self.optimizer_name = optimizer
        self.lr_decay = lr_decay
        self.LSTM_implementation = 2
        self.LSTM_return_sequences = False
        self.mask_zero = True
        self.min_epoch = min_epoch
        self.seed = seed
        self.use_class_weights = use_class_weights
        self.save_folder = save_folder
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH

        # LSTM regularization
        self.LSTM_kernel_regularizer = LSTM_kernel_regularizer
        self.LSTM_kernel_constraint = LSTM_kernel_constraint
        self.LSTM_activity_regularizer = LSTM_activity_regularizer
        self.dropout_LSTM = dropout_LSTM #0.4
        self.recurrent_dropout_LSTM = recurrent_dropout_LSTM #0.4
        self.dense_activity_regularizer = dense_activity_regularizer
        self.trainable_emb = True

        # location path for features
        self.FEATURES_DIR = "%s/data/fnc-1/features/" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

        # param_dict file
        self.PARAM_DICT_FILENAME = param_dict + "_param_dict.pkl"
        print("PARAM_DICT LOADED="+self.PARAM_DICT_FILENAME)

        if self.save_folder is None:
            self.save_folder = "%s/data/fnc-1/mlp_models/temp_models/" % (
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    def __reduce__(self):
        # Called by pickle.dump()
        return (self.__class__, (self.epochs, self.batch_size, "flat_LSTM_100d_50_500", self.lr, "adam", self.lr_decay,
                                 self.dropout_LSTM, self.LSTM_kernel_regularizer, self.LSTM_kernel_constraint,
                                 self.recurrent_dropout_LSTM, self.LSTM_activity_regularizer, self.dense_activity_regularizer,
                                 self.min_epoch, 0.3, self.seed, self.use_class_weights, self.MAX_SEQ_LENGTH, self.save_folder))


    def step_decay(self, epoch):
        """
        Drop LR each epochs_drop by drop_rate.
        :param epoch:
        :return:
        """
        lr = K.get_value(self.model.optimizer.lr)
        new_lr = lr
        if epoch > 0 and float(epoch) % 10 == 0:
            new_lr = float(lr) / 10.0
        #print("\n")
        print("Current LR " + str(lr) + " => New LR " + str(new_lr))
        #print("\n")
        return new_lr


    def fit(self, X_train, y_train, X_test, y_test, test_fold, loss_filename):
        self.test_fold = test_fold
        self.y_test = y_test

        # set session config with gpu variable growth
        self.sess = tf.Session(config=self.config) # see https://github.com/fchollet/keras/issues/1538

        K.set_session(self.sess)

        # convert y_train to one-hot vector
        y_train_one_hot = convert_data_to_one_hot(y_train)
        y_test_one_hot = convert_data_to_one_hot(self.y_test)

        # load feature dict for LSTM_1000_GloVe
        with open(self.FEATURES_DIR+self.PARAM_DICT_FILENAME, "rb") as f:
            param_dict = pickle.load(f)

        # load parameters needed for embedding layer
        EMBEDDING_DIM = param_dict["EMBEDDING_DIM"] # e.g. 50
        self.MAX_SEQ_LENGTH = param_dict["MAX_SEQ_LENGTH"] # e.g. 100

        X_train_LSTM, X_train_MLP = split_X(X_train, self.MAX_SEQ_LENGTH)
        X_test_LSTM, X_test_MLP = split_X(X_test, self.MAX_SEQ_LENGTH)

        # load embeddings
        EMBEDDING_FILE = np.load(self.FEATURES_DIR+param_dict["EMBEDDING_FILE"])

        print("EMBEDDING_FILE.shape = " + str(EMBEDDING_FILE.shape))


        # calc cass weights
        class_weights = calculate_class_weight(y_train, no_classes=4)

        ################
        # CLAIMS LAYER #
        ################
        lstm_input = Input(shape=(self.MAX_SEQ_LENGTH,), dtype='int32', name='lstm_input') # receive sequences of MAX_SEQ_LENGTH_CLAIMS integers
        embedding = Embedding(input_dim=len(EMBEDDING_FILE), # lookup table size
                                    output_dim=EMBEDDING_DIM, # output dim for each number in a sequence
                                    weights=[EMBEDDING_FILE],
                                    input_length=self.MAX_SEQ_LENGTH, # receive sequences of MAX_SEQ_LENGTH_CLAIMS integers
                                    mask_zero=False,
                                    trainable=True)(lstm_input)

        att_vec = GlobalMaxPooling1D()(embedding)

        orig_docs_att = InnerAttentionLayer(att_vec, EMBEDDING_DIM, return_att_weights=True,
                                            return_sequence=True, name='lstm_attention')

        data_LSTM = LSTM(
            100, return_sequences=True, stateful=False, dropout=0.2,
            batch_input_shape=(self.batch_size, self.MAX_SEQ_LENGTH, EMBEDDING_DIM),
            input_shape=(self.MAX_SEQ_LENGTH, EMBEDDING_DIM), implementation=self.LSTM_implementation
            )(orig_docs_att[0])
        data_LSTM = LSTM(
            100, return_sequences=False, stateful=False, dropout=0.2,
            batch_input_shape=(self.batch_size, self.MAX_SEQ_LENGTH, EMBEDDING_DIM),
            input_shape=(self.MAX_SEQ_LENGTH, EMBEDDING_DIM), implementation=self.LSTM_implementation
            )(data_LSTM)

        ###############################
        # MLP (NON-TIMESTEP) FEATURES #
        ###############################
        mlp_input = Input(shape=(len(X_train_MLP[0]),), dtype='float32', name='mlp_input')

        ###############
        # MERGE LAYER #
        ###############
        merged = concatenate([data_LSTM, mlp_input])

        dense_mid = Dense(600, kernel_regularizer=self.regularizer, kernel_initializer=self.kernel_initializer,
                          activity_regularizer=self.dense_activity_regularizer, activation='relu')(merged)
        dense_mid = Dense(600, kernel_regularizer=self.regularizer, kernel_initializer=self.kernel_initializer,
                          activity_regularizer=self.dense_activity_regularizer, activation='relu')(dense_mid)
        dense_mid = Dense(600, kernel_regularizer=self.regularizer, kernel_initializer=self.kernel_initializer,
                          activity_regularizer=self.dense_activity_regularizer, activation='relu')(dense_mid)
        dense_out = Dense(4,activation='softmax', name='dense_out')(dense_mid)

        # build model
        self.model = Model(inputs=[lstm_input, mlp_input], outputs=[dense_out])

        # print summary
        self.model.summary()

        # optimizers
        if self.optimizer_name == "adagrad":
            optimizer = optimizers.Adagrad(lr=self.lr)
            print("Used optimizer: adagrad, lr="+str(self.lr))
        elif self.optimizer_name == "adamax":
            optimizer = optimizers.Adamax(lr=self.lr)  # recommended for sparse stuff like with embeddings
            print("Used optimizer: adamax, lr="+str(self.lr))
        elif self.optimizer_name == "nadam":
            optimizer = optimizers.Nadam(lr=self.lr)  # recommended to leave at default params
            print("Used optimizer: nadam, lr="+str(self.lr))
        elif self.optimizer_name == "rms":
            optimizer = optimizers.RMSprop(lr=self.lr)  # recommended for RNNs
            print("Used optimizer: rms, lr="+str(self.lr))
        elif self.optimizer_name == "SGD":
            optimizer = optimizers.SGD(lr=self.lr)  # recommended for RNNs
            print("Used optimizer: SGD, lr="+str(self.lr))
        elif self.optimizer_name == "adadelta":
            optimizer = optimizers.Adadelta(self.lr)  # recommended for RNNs
            print("Used optimizer: adadelta, lr="+str(self.lr))
        else:
            optimizer = optimizers.Adam(lr=self.lr)
            print("Used optimizer: Adam, lr=" + str(self.lr))

        # compile model: for loss fcts see https://github.com/fchollet/keras/blob/master/keras/losses.py
        self.model.compile(optimizer, 'kullback_leibler_divergence', # categorial_crossentropy
                           metrics=['accuracy']
                           )
        if self.use_class_weights == True:
            self.model.fit([X_train_LSTM, X_train_MLP],
                             y_train_one_hot,
                             validation_data=([X_test_LSTM, X_test_MLP], y_test_one_hot),
                             batch_size=self.batch_size, epochs=self.epochs, verbose=1, class_weight=class_weights,
                           callbacks=[
                               EarlyStoppingOnF1(self.epochs,
                                                 X_test_LSTM, X_test_MLP, self.y_test,
                                                 loss_filename, epsilon=0.0, min_epoch=self.min_epoch),
                           ]
                           )
        else:
            self.model.fit([X_train_LSTM, X_train_MLP],
                             y_train_one_hot,
                             validation_data=([X_test_LSTM, X_test_MLP], y_test_one_hot),
                             batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                           callbacks=[
                               EarlyStoppingOnF1(self.epochs,
                                                 X_test_LSTM, X_test_MLP, self.y_test,
                                                 loss_filename, epsilon=0.0, min_epoch=self.min_epoch),
                           ]
                           )
        self.model.save(self.save_folder+"save.h5")
        return self

    def predict(self, X_test):
        print("Loading model from:" + self.save_folder+"save.h5")
        self.model = load_model(self.save_folder+"save.h5")
        if (self.model != None):
            X_test_LSTM, X_test_MLP = split_X(X_test, self.MAX_SEQ_LENGTH)
            predicted_one_hot = self.model.predict([X_test_LSTM, X_test_MLP])
            predicted_int = np.argmax(predicted_one_hot, axis=-1)

            return predicted_int
        else:
            print("No trained model found.")
            return np.zeros(len(X_test))
