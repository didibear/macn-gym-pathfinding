

from model import MACN
import tensorflow as tf
import time

nb_actions = 4
learning_rate = 0.0001
epochs = 10
batch_size = 128


# DNC Conf
hidden_size = 256
memory_size = 32
word_size = 8
num_reads = 4
num_writes = 1

def main():
    X = tf.placeholder(tf.float32, shape=[None, 9, 9, 2], name='X')
    y = tf.placeholder(tf.int64, shape=[None], name='y') # labels : actions {0,1,2,3}

    logits, prob_actions = MACN(X, k=10, ch_i=2, ch_h=150, ch_q=4, 
        access_config={
            "memory_size": 32, "word_size": 8, "num_reads": 4, "num_writes": 1
        }, 
        controller_config={
            "hidden_size": hidden_size
        },
        batch_size=batch_size
    )

    # Training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
    loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
    train_step = tf.train.RMSPropOptimizer(learning_rate, epsilon=1e-6, centered=True).minimize(loss)

    # Reporting
    y_ = tf.argmax(prob_actions, 1) # predicted action
    nb_errors = tf.reduce_sum(tf.to_float(tf.not_equal(y_, y))) # Number of wrongly selected actions

    def train_batch(X_batch, y_batch):
        _, _nb_errors, _loss = sess.run([train_step, nb_errors, loss], feed_dict={X: X_batch, y: y_batch})
        return _nb_errors, _loss
        
    def test_batch(X_batch, y_batch):
        return sess.run([nb_errors, loss], feed_dict={X: X_batch, y: y_batch})

    trainset = []
    testset = []

    # Start training 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
        for epoch in range(epochs):
            start_time = time.time()
            
            mean_err, mean_loss = compute_on_dataset(sess, trainset, train_batch)
            
            print('Epoch: {:3d} ({:.1f} s):'.format(epoch, time.time() - start_time))
            print('\t Train Loss: {:.5f} \t Train Err: {:.5f}'.format(mean_loss, mean_err))

            saver.save(sess, "./model/weights.ckpt")

        print('Training finished')

        print('Testing...')

        mean_err, mean_loss = compute_on_dataset(sess, testset, test_batch)

        print('Test Accuracy: {:.2f}%'.format(100*(1 - mean_err)))



def compute_on_dataset(sess, dataset, compute_batch):
    nb_batches = dataset.size // batch_size
    nb_examples = nb_batches * batch_size

    total_errors = 0.0
    total_loss = 0.0

    for batch in range(nb_batches):
        X_batch, y_batch = dataset.next_batch(batch_size)
        
        nb_errors, loss = compute_batch(X_batch, y_batch)
            
        total_errors += nb_errors
        total_loss += loss

    mean_error = total_errors / nb_examples
    mean_loss = total_loss / nb_examples
    return mean_error, mean_loss



if __name__ == "__main__":
    main()