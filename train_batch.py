import tensorflow as tf
import time
import numpy as np
import os

from model import BatchMACN, VINConfig
from dataset import get_datasets

FLAGS = tf.flags.FLAGS

# Hyperparameter
tf.flags.DEFINE_integer("epochs",           30,    "Number of epochs for training")
tf.flags.DEFINE_integer("batch_per_epoch",  100,   "Number of episodes per epochs")
tf.flags.DEFINE_float(  "learning_rate",    10e-5, "The learning rate")

# MACN conf
tf.flags.DEFINE_integer("im_h", 9,  "Image height")
tf.flags.DEFINE_integer("im_w", 9,  "Image width")
tf.flags.DEFINE_integer("ch_i", 2,  "Channels in input layer (~2 in [grid, reward])")

# Batch MACN conf
tf.flags.DEFINE_integer("batch_size",   32,  "Batch size (batch of episode)")
tf.flags.DEFINE_integer("seq_length",   10, "Length of an episode (nb timesteps)")

# VIN conf
tf.flags.DEFINE_integer("k",    10,     "Number of iteration for planning (VIN)")
tf.flags.DEFINE_integer("ch_q", 4,      "Channels in q layer (~actions)")
tf.flags.DEFINE_integer("ch_h", 150,    "Channels in initial hidden layer")

# DNC Conf
tf.flags.DEFINE_integer("hidden_size",      256,    "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size",      32,     "The number of memory slots.")
tf.flags.DEFINE_integer("word_size",        8,      "The width of each memory slot.")
tf.flags.DEFINE_integer("num_read_heads",   4,      "Number of memory read heads.")
tf.flags.DEFINE_integer("num_write_heads",  1,      "Number of memory write heads.")

tf.flags.DEFINE_string('dataset', "./data/dataset.pkl", "Path to dataset file")
tf.flags.DEFINE_string('save', "./model/weights_batch.ckpt", "File to save the weights")
tf.flags.DEFINE_string('load', "./model/weights_batch.ckpt", "File to load the weights")

def main(args):
    checks()

    macn = BatchMACN(
        image_shape=[FLAGS.im_h, FLAGS.im_w, FLAGS.ch_i],
        vin_config=VINConfig(k=FLAGS.k, ch_h=FLAGS.ch_h, ch_q=FLAGS.ch_q),
        access_config={
            "memory_size": FLAGS.memory_size, 
            "word_size": FLAGS.word_size, 
            "num_reads": FLAGS.num_read_heads, 
            "num_writes": FLAGS.num_write_heads
        }, 
        controller_config={
            "hidden_size": FLAGS.hidden_size
        }, 
        batch_size=FLAGS.batch_size,
        seq_length=FLAGS.seq_length        
    )

    # y = [batch, labels]
    y = tf.placeholder(tf.int64, shape=[None, None], name='y') # labels : actions {0,1,2,3}

    # Training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=macn.logits, name='cross_entropy')
    loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
    train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate, epsilon=1e-6, centered=True).minimize(loss)

    # Reporting
    y_ = tf.argmax(macn.prob_actions, axis=-1) # predicted action
    nb_errors = tf.reduce_sum(tf.to_float(tf.not_equal(y_, y))) # Number of wrongly selected actions

    def train_on_episode_batch(batch_images, batch_labels):
        _, _loss, _nb_err = sess.run([train_step, loss, nb_errors], feed_dict={macn.X : batch_images, y : batch_labels})
        return _loss, _nb_err
        
    def test_on_episode_batch(batch_images, batch_labels):
        return sess.run([loss, nb_errors], feed_dict={macn.X : batch_images, y : batch_labels})

    trainset, testset = get_datasets(FLAGS.dataset, test_percent=0.1)
    
    # Start training 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if loadfile_exists(FLAGS.load):
            saver.restore(sess, FLAGS.load)
            print("Weights reloaded")
        else:
            sess.run(tf.global_variables_initializer())
       
        print("Start training...")
        for epoch in range(1, FLAGS.epochs + 1):
            start_time = time.time()

            mean_loss, mean_accuracy = compute_on_dataset(sess, trainset, train_on_episode_batch)
            
            print('Epoch: {:3d} ({:.1f} s):'.format(epoch, time.time() - start_time))
            print('\t Train Loss: {:.5f} \t Train accuracy: {:.2f}%'.format(mean_loss, 100*(mean_accuracy)))

            saver.save(sess, FLAGS.save)
        print('Training finished.')


        print('Testing...')
        mean_loss, mean_accuracy = compute_on_dataset(sess, testset, test_on_episode_batch)
        print('Test Accuracy: {:.2f}%'.format(100*(mean_accuracy)))


def compute_on_dataset(sess, dataset, compute_episode_batch):
    total_loss = 0
    total_accuracy = 0

    for batch in range(1, FLAGS.batch_per_epoch + 1):
        
        batch_images, batch_labels = dataset.next_episode_batch(FLAGS.batch_size)
        
        loss, nb_err = compute_episode_batch(batch_images, batch_labels)

        accuracy = 1 - (nb_err / (FLAGS.batch_size * FLAGS.seq_length))

        total_loss += loss / FLAGS.batch_size
        total_accuracy += accuracy
    
    mean_loss = total_loss / FLAGS.batch_per_epoch
    mean_accuracy = total_accuracy / FLAGS.batch_per_epoch
    return mean_loss, mean_accuracy

        
def loadfile_exists(filepath):
    filename = os.path.basename(filepath)
    for file in os.listdir(os.path.dirname(filepath)):
        if file.startswith(filename):
            return True
    return False

def checks():
    if not os.path.exists(os.path.dirname(FLAGS.save)):
        print("Error : save file cannot be created (need folders) : " + FLAGS.save)
        exit()



if __name__ == "__main__":
    tf.app.run()