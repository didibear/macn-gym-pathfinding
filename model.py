import tensorflow as tf
from  dnc.dnc import DNC

import numpy as np
import tensorflow as tf

from collections import namedtuple

VINConfig = namedtuple("VINConf", ["k", "ch_h", "ch_q"])

class MACN(object):
    """
    Parameter
    ---------
    image_shape: the shape of the input image : [m, n, ch_i] (type list)
        - m : number of lines
        - n : number of columns
        - ch_i : channels (=2 in [grid, reward])
    vin_config: the VIN config (type VINConfig)
        - k : Number of Value Iteration computations
        - ch_h : Channels in initial hidden layer
        - ch_q : Channels in q layer (~actions)
    access_config: dictionary of access module configurations.
        (memory_size, word_size, num_reads, num_writes, name)
    controller_config: dictionary of controller (LSTM) module configurations.
        (hidden_size)

    Fields
    ------
    X : ([seq_length, m, n, ch_i) - episode with images (stack of gridworld and goal)
        - gridworld (m, n) = grid with 1 and 0 ;
        - goal (m, n) = grid with 10 at goal position
    prob_actions : (ch_q,) - the action probabilities, execute argmax on it to know the action
    """
    def __init__(self, *, image_shape, vin_config, access_config, controller_config):
        
        self.X = tf.placeholder(tf.float32, shape=[None] + image_shape, name='X')

        # Execute planning (estimate state value function)
        self.v = VIN(self.X, vin_config)

        v_batch = tf.expand_dims(self.v, axis=0, name="v_batch")

        # Memory part
        self.dnc_core = DNC(access_config, controller_config, output_size=vin_config.ch_q)
        self.state_in = self.dnc_core.initial_state(1)

        self.logits, self.state_out = tf.nn.dynamic_rnn(
            cell=self.dnc_core,
            inputs=v_batch,
            initial_state=self.state_in
        )

        # Remove the unused batch dimension
        self.logits = tf.squeeze(self.logits, axis=[0])

        self.prob_actions = tf.nn.softmax(self.logits, name='probability_actions')


class BatchMACN(object):
    """
    Parameter
    ---------
    (see MACN docstring)
    batch_size : the batch size (input in X).
    seq_length : the sequence length of an episode (input in X).

    Fields
    ------
    X : (batch_size, seq_length, m, n, ch_i) - episodes with images (stack of gridworld and goal)
        - gridworld (m, n) = grid with 1 and 0 ;
        - goal (m, n) = grid with 10 at goal position
    prob_actions : (batch_size, ch_q,) - the action probabilities, execute argmax() on it to know the action

    """
    def __init__(self, *, image_shape, vin_config, access_config, controller_config, batch_size, seq_length):
        
        self.X = tf.placeholder(tf.float32, shape=[batch_size, seq_length] + image_shape, name='X')
        
        # Merge batch and sequence dims for conv2d
        X_unbatch = tf.reshape(self.X, shape=[batch_size * seq_length] + image_shape)

        # Execute planning (estimate state value function)
        self.v = VIN(X_unbatch, vin_config)

        v_batch = tf.reshape(self.v, shape=[batch_size, seq_length, -1])

        # Memory part
        self.dnc_core = DNC(access_config, controller_config, output_size=vin_config.ch_q)
        self.state_in = self.dnc_core.initial_state(batch_size)

        self.logits, self.state_out = tf.nn.dynamic_rnn(
            cell=self.dnc_core,
            inputs=v_batch,
            initial_state=self.state_in
        )
        self.prob_actions = tf.nn.softmax(self.logits, name='probability_actions')



def VIN(X, vin_config):
    """
    Value Iteration Network

    X : (?, m, n, ch_i) - batch of images (stack of gridworld and goal)
        - gridworld (m, n) = grid with 1 and 0 ;
        - goal (m, n) = grid with 10 at goal position
    vin_config: the VIN config (type VINConfig)
        - k: Number of Value Iteration computations
        - ch_h : Channels in initial hidden layer
        - ch_q : Channels in q layer (~actions)
    """

    h = conv2d(inputs=X, filters=vin_config.ch_h, name='h0', use_bias=True)
    r = conv2d(inputs=h, filters=1, name='r')
            
    # Initialize value map (zero everywhere)
    v = tf.zeros_like(r)

    rv = tf.concat([r, v], axis=3)
    q = conv2d(inputs=rv, filters=vin_config.ch_q, name='q', reuse=None)  # Initial set before sharing weights
    v = tf.reduce_max(q, axis=3, keep_dims=True, name='v')

    # K iterations of VI module
    for _ in range(vin_config.k):
        rv = tf.concat([r, v], axis=3)
        q = conv2d(inputs=rv, filters=vin_config.ch_q, name='q', reuse=True) # Sharing weights
        v = tf.reduce_max(q, axis=3, keep_dims=True, name='v')

    return v

def conv2d(*, inputs, filters, name, use_bias=False, reuse=False):
    return tf.layers.conv2d(
        inputs=inputs, 
        filters=filters, 
        kernel_size=[3, 3], 
        strides=[1, 1], 
        padding='same', 
        activation=None, 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
        use_bias=use_bias,
        bias_initializer=tf.zeros_initializer() if use_bias else None,
        name=name,
        reuse=reuse
    )
