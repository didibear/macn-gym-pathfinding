import tensorflow as tf
import time
import numpy as np

from model import MACN
from dataset import get_datasets

# Resume
load = True

# Hyperparameter
nb_actions = 4
learning_rate = 0.0001
epochs = 10
episodes_per_epoch = 1000
batch_size = 128
report_interval = 1000

imsize = 7

### MACN conf

# VIN conf
k = 10
ch_i = 2
ch_h = 150
ch_q = 4

# DNC conf
hidden_size = 256
memory_size = 32
word_size = 8
num_reads = 4
num_writes = 1


def main():
    X = tf.placeholder(tf.float32, shape=[None, imsize, imsize, 2], name='X')
    y = tf.placeholder(tf.int64, shape=[None], name='y') # labels : actions {0,1,2,3}

    macn = MACN(X, k=k, ch_i=ch_i, ch_h=ch_h, ch_q=ch_q, 
        access_config={
            "memory_size": memory_size, "word_size": word_size, "num_reads": num_reads, "num_writes": num_writes
        }, 
        controller_config={
            "hidden_size": hidden_size
        }
    )

    # Training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=macn.logits, name='cross_entropy')
    loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
    train_step = tf.train.RMSPropOptimizer(learning_rate, epsilon=1e-6, centered=True).minimize(loss)

    # Reporting
    y_ = tf.argmax(macn.prob_actions, 1) # predicted action
    nb_errors = tf.reduce_sum(tf.to_float(tf.not_equal(y_, y))) # Number of wrongly selected actions

    def train_on_episode(images, labels):
        _, _loss, _nb_err = sess.run([train_step, loss, nb_errors], feed_dict={X : images, y : labels})
        return _loss, _nb_err
        
    def test_on_episode(images, labels):
        return sess.run([loss, nb_errors], feed_dict={X : images, y : labels})

    trainset, testset = get_datasets("./data/dataset.pkl", test_percent=0.1)
    
    # Start training 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if (load):
            saver.restore(sess, "./model/weights.ckpt")
        else:
            sess.run(tf.global_variables_initializer())
       
        print("Start training...")
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            mean_loss, mean_accuracy = compute_on_dataset(sess, trainset, train_on_episode)
            
            print('Epoch: {:3d} ({:.1f} s):'.format(epoch, time.time() - start_time))
            print('\t Train Loss: {:.5f} \t Train accuracy: {:.2f}%'.format(mean_loss, 100*(mean_accuracy)))

            saver.save(sess, "./model/weights.ckpt")
        print('Training finished.')

        print('Testing...')
        mean_loss, mean_accuracy = compute_on_dataset(sess, testset, test_on_episode)
        print('Test Accuracy: {:.2f}%'.format(100*(mean_accuracy)))

def compute_on_dataset(sess, dataset, compute_episode):
    total_loss = 0
    total_accuracy = 0

    for episode in range(1, episodes_per_epoch + 1):
        # model_state = macn.dnc_core.zero_state(1, dtype=tf.float32)
    
        images, labels = dataset.next_episode()
        
        # images = np.expand_dims(images, axis=0) # simulate batch
        # labels = np.expand_dims(labels, axis=0) # simulate batch
        # sess.reset(macn.state_in)

        loss, nb_err = compute_episode(images, labels)
        
        accuracy = 1 - (nb_err / labels.shape[0])
        
        total_loss += loss
        total_accuracy += accuracy
    
    mean_loss = total_loss / episodes_per_epoch
    mean_accuracy = total_accuracy / episodes_per_epoch
    return mean_loss, mean_accuracy



if __name__ == "__main__":
    main()