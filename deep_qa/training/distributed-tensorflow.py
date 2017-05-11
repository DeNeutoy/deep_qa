import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ["host:port"], "worker": ["host:port"]})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        # Workers do all the work - parameter servers are just there
        # to host the variables. So we just make it wait until the
        # server joins, doing absolutely jack.
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        # Assigns variables to a parameter server, defaulting to ps.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # input images
            with tf.name_scope('input'):
                # None -> batch size can be any size, 784 -> flattened mnist image
                inputs = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
                # target 10 output classes
                targets = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

            # model parameters will change during training so we use tf.Variable
            tf.set_random_seed(1)
            with tf.name_scope("weights"):
                W1 = tf.Variable(tf.random_normal([784, 100]))
                W2 = tf.Variable(tf.random_normal([100, 10]))

            # bias
            with tf.name_scope("biases"):
                b1 = tf.Variable(tf.zeros([100]))
                b2 = tf.Variable(tf.zeros([10]))

            # implement model
            with tf.name_scope("softmax"):
                # y is our prediction
                layer1 = tf.add(tf.matmul(inputs, W1), b1)
                layer1 = tf.nn.sigmoid(layer1)
                layer2 = tf.add(tf.matmul(layer1, W2), b2)
                predictions = tf.nn.softmax(layer2)

            # specify cost function
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(predictions), reduction_indices=[1]))

            # Integer variable which counts gradient updates across hosts.
            global_step = tf.train.get_or_create_global_step()

            # Special optimiser wrapper which will collect gradients
            # and then average them before applying them.
            optimiser = tf.train.SyncReplicasOptimizer(
                tf.train.AdagradOptimizer(0.01), 3)
            train_op = optimiser.minimize(loss, global_step=global_step)

        sync_replicas_hook = optimiser.make_session_run_hook(FLAGS.task_index==0)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000), sync_replicas_hook]

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        batch_size = 64

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks) as monitored_session:
            while not monitored_session.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                batch_inputs, batch_targets = mnist.train.next_batch(batch_size)
                _, cost, step = monitored_session.run([train_op, loss, global_step],
                                                      feed_dict={inputs: batch_inputs, targets: batch_targets})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument("--job_name", type=str,
                        default="", help="One of 'ps', 'worker'")
    # Flags for defining the tf.train.Server
    parser.add_argument("--task_index", type=int,
                        default=0, help="Index of task within the job")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
