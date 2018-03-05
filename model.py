import tensorflow as tf
from  dnc.dnc import DNC

import numpy as np
import tensorflow as tf


class MACN(object):
    def __init__(self, X, *, k, ch_i, ch_h, ch_q, access_config, controller_config, batch_size=1):
        """
        X : (timesteps, m, n, 2) - episode with stack of gridworld and goal
            gridworld (m, n) = grid with 1 and 0 ;
            goal (m, n) = grid with 10 at goal position
            
        k: Number of Value Iteration computations
        ch_i: Channels in input layer
        ch_h: Channels in initial hidden layer
        ch_q: Channels in q layer (~actions)
        access_config: dictionary of access module configurations.
            (memory_size, word_size, num_reads, num_writes, name)
        controller_config: dictionary of controller (LSTM) module configurations.
            (hidden_size)
        """
        self.batch_size = 1
        CHANNEL_AXIS = 3

        self.h = conv2d(inputs=X, filters=ch_h, name='h0', use_bias=True)
        self.r = conv2d(inputs=self.h, filters=1, name='r')
               
        # Initialize value map (zero everywhere)
        self.v = tf.zeros_like(self.r)

        self.rv = tf.concat([self.r, self.v], axis=CHANNEL_AXIS)
        self.q = conv2d(inputs=self.rv, filters=ch_q, name='q', reuse=None)  # Initial set before sharing weights
        self.v = tf.reduce_max(self.q, axis=CHANNEL_AXIS, keep_dims=True, name='v')

        # K iterations of VI module
        for _ in range(k):
            self.rv = tf.concat([self.r, self.v], axis=CHANNEL_AXIS)
            self.q = conv2d(inputs=self.rv, filters=ch_q, name='q', reuse=True) # Sharing weights
            self.v = tf.reduce_max(self.q, axis=CHANNEL_AXIS, keep_dims=True, name='v')

        self.v_batch = tf.expand_dims(self.v, 0)

        # DNC Net   
        self.dnc_core = DNC(access_config, controller_config, output_size=ch_q)

        self.state_in = self.dnc_core.initial_state(self.batch_size)
    
        self.logits, self.state_out = tf.nn.dynamic_rnn(
            cell=self.dnc_core,
            inputs=self.v_batch,
            initial_state=self.state_in
            # dtype=tf.float32
        )

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
