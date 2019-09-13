import numpy as np 
import os
import tensorflow as tf 
import time, datetime
from trash_cnn import Trash_CNN
from data_process import load_data, batch_iter

# Data loading params
tf.flags.DEFINE_string("training", "./garbage-classification/one-indexed-files-notrash_train.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("validation", "./garbage-classification/one-indexed-files-notrash_val.txt", "Data source for the negative data.")
tf.flags.DEFINE_string("testing", "./garbage-classification/one-indexed-files-notrash_test.txt", "Data source for the negative data.")

# Training parameters
tf.flags.DEFINE_integer("num_classes", 6, " Num Classes (default: 6)")
tf.flags.DEFINE_string("filter_sizes", "5, 3, 3", "Comma-separated filter sizes (default: '5, 3, 3')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters initial (default: 64)")
tf.flags.DEFINE_integer("input_size", 256, "Input size for images (default: 256)")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def train():

    print("Loading data...")

    train_data, val_data, test_data = load_data(FLAGS.training, FLAGS.validation, FLAGS.testing)
    
    print("Train {}".format(np.array(train_data).shape))

    print("Initializing...")

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)
        
        with sess.as_default():

            cnn = Trash_CNN(
                num_classes=FLAGS.num_classes, 
                input_shape=(FLAGS.input_size, FLAGS.input_size, 3), 
                filters=list(map(int, FLAGS.filter_sizes.split(","))), 
                input_channel=FLAGS.num_filters)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)

            train_op = optimizer.apply_gradients(grads_and_vars)

            timestamp = str(int(time.time()))
            outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(outdir))

            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("acc", cnn.accuracy)
            
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(outdir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(outdir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(outdir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            sess.run(tf.global_variables_initializer())

            def train_step(batch_x, batch_y):
                '''
                    One single training step
                '''

                feed_dict = {
                    cnn.input_x: batch_x,
                    cnn.input_y: batch_y
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict=feed_dict
                )

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc: {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(batch_x, batch_y, writer=None):
                '''
                    Evaluate model
                '''

                feed_dict = {
                    cnn.input_x: batch_x,
                    cnn.input_y: batch_y
                }

                step, summaries, loss, accuracy = sess.run(
                    [global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict=feed_dict
                )

                time_str = datetime.datetime.now().isoformat()
                print("-------------- Step summary ---------------")
                print("{}: step {}, loss {:g}, acc: {:g}".format(time_str, step, loss, accuracy))

                if writer:
                    writer.add_summary(summaries, step)

            batches = batch_iter(
                list(zip(train_data[0], train_data[1])), 
                FLAGS.batch_size, 
                FLAGS.num_epochs
            )

            for batch in batches:

                print("Batch {}".format(batch.shape))
                x_batch, y_batch = zip(*batch)

                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\n======================= Evaluation: =======================")
                    dev_step(val_data[0], val_data[1], writer=dev_summary_writer)
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
