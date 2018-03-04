import tensorflow as tf
from  dnc.dnc import DNC

import numpy as np
import tensorflow as tf

def MACN(X, *, k, ch_i, ch_h, ch_q, access_config, controller_config, batch_size):
    """
    X : (?, m, n, 2) - stack of gridworld and goal
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
    h = conv2d(inputs=X, filters=ch_h, name='h0', use_bias=True)
    r = conv2d(inputs=h, filters=1, name='r')
    
    # Add collection of reward image
    tf.add_to_collection('r', r)
    
    # Initialize value map (zero everywhere)
    v = tf.zeros_like(r)

    rv = tf.concat([r, v], axis=3)
    q = conv2d(inputs=rv, filters=ch_q, name='q', reuse=None)  # Initial set before sharing weights
    v = tf.reduce_max(q, axis=3, keep_dims=True, name='v')

    # K iterations of VI module
    for _ in range(k):
        rv = tf.concat([r, v], axis=3)
        q = conv2d(inputs=rv, filters=ch_q, name='q', reuse=True) # Sharing weights
        v = tf.reduce_max(q, axis=3, keep_dims=True, name='v')

    # Add collection of value images
    tf.add_to_collection('v', v)  

    # DNC Net   

    dnc_core = DNC(access_config, controller_config, output_size=ch_q)
    initial_state = dnc_core.initial_state(batch_size)

    logits, _ = tf.nn.dynamic_rnn(
        cell=dnc_core,
        inputs=v,
        time_major=True,
        initial_state=initial_state
    )

    prob_actions = tf.nn.softmax(logits, name='probability_actions')

    return logits, prob_actions

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
