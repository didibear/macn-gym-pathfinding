

import numpy as np
import tensorflow as tf
from collections import namedtuple

from macn.dnc.dnc import DNC

class MACNConfig(namedtuple("MACNConfig", ["image_shape", "vin_config", "access_config", "controller_config"])):
    """
    ### Configuration for MACN model
    image_shape :   The shape of the input image (e.g. [9, 9, 2])

    ### VIN conf
    vin_config : dictionnary of the VIN config
        - k :       Number of iteration for planning (value iteration)
        - ch_q :    Channels in q layer (~actions)
        - ch_h :    Channels in initial hidden layer

    ### DNC Conf
    access_config : dictionary of access module configurations.
        - memory_size :     The number of memory slots.
        - word_size :       The width of each memory slot.
        - num_read_heads :  Number of memory read heads.
        - num_write_heads : Number of memory write heads.
    controller_config : dictionary of controller (LSTM) module configurations.
        - hidden_size :     Size of LSTM hidden layer.
    """
    pass

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
    @classmethod
    def from_spec(cls, spec):
        """ spec: the model spec (type MACNConfig) """
        return cls(
            image_shape=spec.image_shape,
            vin_config=VINConfig(**spec.vin_config),
            access_config=spec.access_config, 
            controller_config=spec.controller_config
        )

    def __init__(self, *, image_shape, vin_config, access_config, controller_config):
        
        self.X = tf.placeholder(tf.float32, shape=[None] + image_shape, name='X')

        # Execute planning (estimate state value function)
        self.rv = VIN(self.X, vin_config)

        rv_batch = tf.expand_dims(self.rv, axis=0, name="rv_batch")

        # Memory part
        self.dnc_core = DNC(access_config, controller_config, output_size=vin_config.ch_q)
        self.state_in = self.dnc_core.initial_state(1)

        self.logits, self.state_out = tf.nn.dynamic_rnn(
            cell=self.dnc_core,
            inputs=rv_batch,
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
    @classmethod
    def from_spec(cls, spec, batch_size, seq_length):
        """ spec: the model spec (type MACNConfig) """
        return cls(
            image_shape=spec.image_shape,
            vin_config=VINConfig(**spec.vin_config),
            access_config=spec.access_config, 
            controller_config=spec.controller_config,
            batch_size=batch_size, 
            seq_length=seq_length
        )

    def __init__(self, *, image_shape, vin_config, access_config, controller_config, batch_size, seq_length):
        
        self.X = tf.placeholder(tf.float32, shape=[batch_size, seq_length] + image_shape, name='X')
        
        # Merge batch and sequence dims for conv2d
        X_unbatch = tf.reshape(self.X, shape=[batch_size * seq_length] + image_shape)

        # Execute planning (estimate state value function)
        self.rv = VIN(X_unbatch, vin_config)

        rv_batch = tf.reshape(self.rv, shape=[batch_size, seq_length, -1], name="rv_batch")

        # Memory part
        self.dnc_core = DNC(access_config, controller_config, output_size=vin_config.ch_q)
        self.state_in = self.dnc_core.initial_state(batch_size)

        self.logits, self.state_out = tf.nn.dynamic_rnn(
            cell=self.dnc_core,
            inputs=rv_batch,
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
    v = tf.reduce_max(q, axis=3, keepdims=True, name='v')

    # K iterations of VI module
    for _ in range(vin_config.k):
        rv = tf.concat([r, v], axis=3)
        q = conv2d(inputs=rv, filters=vin_config.ch_q, name='q', reuse=True) # Sharing weights
        v = tf.reduce_max(q, axis=3, keepdims=True, name='v')

    rv = tf.concat([r, v], axis=3)
    return rv

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

