from keras import initializers, regularizers, constraints, activations
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class InnerAttentionLayer(Layer):
    #http://bingning.wang/site_media/download/ACL2016.pdf
    def __init__(self, topic, emb_dim=300, return_sequence=False, W_regularizer=None, W_constraint=None, return_att_weights=False,
                 **kwargs):

        self.supports_masking = True
        self.init = initializers.glorot_uniform()

        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)

        self.emb_dim = emb_dim
        self.topic = topic
        self.return_sequences = return_sequence
        self.return_att_weights = return_att_weights

        super(InnerAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(self.topic.get_shape().as_list()[-1], self.emb_dim,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        super(InnerAttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):  # x = (bs, 50, 100), topic = (bs, 100)

        # W * s
        s_flat = K.reshape(x, [-1, K.shape(x)[2]])  # s_flat = (bs*50, 100)
        W_s_topic_flat = K.transpose(dot_product(self.W, s_flat))  # transpose((100, 100) * (100, bs*50)) => tp(100, bs*50) => (bs*50, 100)
        W_s = tf.reshape(W_s_topic_flat, [K.shape(x)[0], K.shape(x)[1], -1])  # (bs, 50, 100)

        # t * W_s
        t = K.expand_dims(self.topic, axis=1)  # t= (bs, 1, 100)
        # t_W_s = dot_product(t, K.transpose(K.permute_dimensions(W_s, (0, 2, 1)))) # (bs, 100, 50)
        W_s_transpose = tf.transpose(W_s, perm=[0, 2, 1])
        t_W_s = tf.matmul(t, W_s_transpose)  # (bs, 1, 100) * (bs, 100, 50) = (bs, 1, 50)
        t_W_s = K.squeeze(t_W_s, axis=1)  # (bs, 50)

        a = K.sigmoid(t_W_s)  # (bs, 50)

        # weight lstm states with alphas
        attention_weights = K.expand_dims(a)  # (bs, 50, 1)
        weighted_states = x * attention_weights  # (bs, 50, 100) * (bs, 50, 1)

        if self.return_sequences:
            final_states =  weighted_states
        else:
            final_states =  K.sum(weighted_states, axis=1)

        if self.return_att_weights == False:
            return final_states
        else:
            return [final_states, a]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        if self.return_sequences:
            final_states_shape = tuple([input_shape[0], input_shape[1], input_shape[2]])
        else:
            final_states_shape = tuple([input_shape[0], input_shape[2]])

        if self.return_att_weights == False:
            return final_states_shape
        else:
            return [final_states_shape, tuple([input_shape[0], input_shape[1]])]

