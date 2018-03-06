import tensorflow as tf
from  dnc.dnc import DNC

import numpy as np
import tensorflow as tf


class MACN(object):
    def __init__(self, *, image_shape, vin_config, access_config, controller_config, batch_size=None, seq_length=None):
        """
        image_shape: the shape of the input image : [m, n, ch_i] (type list)
            - m : number of lines
            - n : number of columns
            - ch_i : channels (=2 in [grid,reward])
        vin_config: the VIN config
            - k : Number of Value Iteration computations
            - ch_h : Channels in initial hidden layer
            - ch_q : Channels in q layer (~actions)
        access_config: dictionary of access module configurations.
            (memory_size, word_size, num_reads, num_writes, name)
        controller_config: dictionary of controller (LSTM) module configurations.
            (hidden_size)
        batch_size : the batch size. If set, the batch dims have to be present in X.
        seq_length : the batch size. If set, the batch dims have to be present in X.

        Fields
        ------
        X : ([batch_size,] seq_length, m, n, ch_i) - episode with images (stack of gridworld and goal)
            - gridworld (m, n) = grid with 1 and 0 ;
            - goal (m, n) = grid with 10 at goal position
        """
        if batch_size: assert seq_length
        if seq_length: assert batch_size
    
        k = vin_config["k"]
        ch_h = vin_config["ch_h"]
        ch_q = vin_config["ch_q"]

        if batch_size: 
            self.X_batch = tf.placeholder(tf.float32, shape=[batch_size, seq_length] + image_shape, name='X_batch')
            # Reshape to merge batch and sequence dims
            self.X = tf.reshape(self.X_batch, shape=[batch_size * seq_length] + image_shape, name="X")
        else:
            self.X = tf.placeholder(tf.float32, shape=[None] + image_shape, name='X')

        ### VIN Model
        self.h = conv2d(inputs=self.X, filters=ch_h, name='h0', use_bias=True)
        self.r = conv2d(inputs=self.h, filters=1, name='r')
               
        # Initialize value map (zero everywhere)
        self.v = tf.zeros_like(self.r)

        self.rv = tf.concat([self.r, self.v], axis=3)
        self.q = conv2d(inputs=self.rv, filters=ch_q, name='q', reuse=None)  # Initial set before sharing weights
        self.v = tf.reduce_max(self.q, axis=3, keep_dims=True, name='v')

        # K iterations of VI module
        for _ in range(k):
            self.rv = tf.concat([self.r, self.v], axis=3)
            self.q = conv2d(inputs=self.rv, filters=ch_q, name='q', reuse=True) # Sharing weights
            self.v = tf.reduce_max(self.q, axis=3, keep_dims=True, name='v')

        # Reshape to disjoin batch and sequence dims
        if batch_size: 
            # v_s = tf.shape(self.v)
            # assert_shape = tf.Assert(X_s[0] == batch_size * seq_length, ['v shape and batch_size/seq_length have to be equals'])
            # shape = [batch_size, seq_length, v_s[1], v_s[2], v_s[3]]
            self.v_batch = tf.reshape(self.v, shape=[batch_size, seq_length, -1], name="v_batch")
            # self.v_batch.set_shape(shape)
            # self.v_batch = tf.control_flow_ops.with_dependencies([assert_shape], self.v_batch)
        else:
            self.v_batch = tf.expand_dims(self.v, axis=0, name="v_batch")
            
        ### DNC Net   
        self.dnc_core = DNC(access_config, controller_config, output_size=ch_q)

        self.state_in = self.dnc_core.initial_state(batch_size or 1)
    
        self.logits, self.state_out = tf.nn.dynamic_rnn(
            cell=self.dnc_core,
            inputs=self.v_batch,
            initial_state=self.state_in
        )

        if not batch_size:
            # Remove the first dimension
            self.logits = tf.squeeze(self.logits, axis=[0])

        self.prob_actions = tf.nn.softmax(self.logits, name='probability_actions')

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
