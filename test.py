import tensorflow as tf


inputs = tf.constant([[
    [
        [[1,2,3],
         [2,2,3],
         [1,3,3]],

        [[8,2,3],
         [1,4,3],
         [7,2,3]],
    ],
    [
        [[4,5,6],
         [4,5,6],
         [4,5,6]],

        [[4,5,6],
         [4,5,6],
         [4,5,6]],  
    ]
]], dtype=tf.float32)

s = tf.shape(inputs)
c = tf.layers.conv2d(
    inputs=inputs, 
    filters=1, 
    kernel_size=[3,3], 
    strides=[1, 1], 
    padding='same', 
    activation=None, 
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
    name="mdr",
    reuse=False
)

sc = tf.shape(c)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([inputs]))
    print(sess.run([s]))
    print(sess.run([c]))
    print(sess.run([sc]))
    

