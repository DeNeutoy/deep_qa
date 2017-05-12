import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ["mark-new-cuda.dev.ai2:4004"],
                                    "worker": ["mark-new-cuda.dev.ai2:1234", "mark-new-cuda.dev.ai2:1235"]})

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

        # Chief worker is always the first one.
        is_chief = FLAGS.task_index == 0

        # Assigns ops to the local worker by default.
        # Assigns variables to a parameter server, defaulting to ps.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}/gpu:{}".format(FLAGS.task_index, FLAGS.task_index),
                cluster=cluster)):

            # input images
            with tf.name_scope('input'):
                # None -> batch size can be any size, 784 -> flattened mnist image
                inputs = tf.placeholder(tf.float32, shape=[None, 784], name="inputs")
                # target 10 output classes
                targets = tf.placeholder(tf.float32, shape=[None, 10], name="targets")

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

            tf.summary.scalar("loss", loss)

            # Integer variable which counts gradient updates across hosts.
            global_step = tf.train.get_or_create_global_step()

            # Special optimiser wrapper which will collect gradients
            # and then average them before applying them.
            optimiser = tf.train.SyncReplicasOptimizer(
                tf.train.AdagradOptimizer(0.01), 3)
            train_op = optimiser.minimize(loss, global_step=global_step)

            summary_op = tf.summary.merge_all()

            ### Initialisation operations for the graph. ###

            variable_init_op = tf.global_variables_initializer()

            # You can now call get_init_tokens_op() and get_chief_queue_runner().
            # Note that get_init_tokens_op() must be called before creating session
            # because it modifies the graph.
            local_init_op = optimiser.local_step_init_op

            # Chief has a different init op, as it does more things.
            if is_chief:
                local_init_op = optimiser.chief_init_op
            ready_for_local_init_op = optimiser.ready_for_local_init_op
            # Initial token and chief queue runners required by the sync_replicas mode
            chief_queue_runner = optimiser.get_chief_queue_runner()
            sync_init_op = optimiser.get_init_tokens_op()

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=1000000)]

            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            batch_size = 64
            max_steps = 10000

            # Now we create a Supervisor, which ties together some stuff which we need for training,
            # like initialisation of the workers, finalising the graph etc.

            saver = tf.train.Saver()
            supervisor = tf.train.Supervisor(
                is_chief=is_chief,
                logdir="./log",
                saver=saver,
                summary_op=summary_op,
                init_op=variable_init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step)

            if is_chief:
                print("Worker %d: Initializing session..." % FLAGS.task_index)
            else:
                print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)

            session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

            with supervisor.prepare_or_wait_for_session(server.target, config=session_config) as session:

                if is_chief:
                    # Chief worker will start the chief queue runner and call the init op.
                    session.run(sync_init_op)
                    supervisor.start_queue_runners(session, [chief_queue_runner])

                step = 0
                while step < max_steps:
                    # Run a training step asynchronously.
                    # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                    # perform *synchronous* training.

                    batch_inputs, batch_targets = mnist.train.next_batch(batch_size)
                    _, cost, step = session.run([train_op, loss, global_step],
                                                feed_dict={inputs: batch_inputs, targets: batch_targets})

                    print("Worker: {}, Step: {},  Training Loss: {}".format(FLAGS.task_index, step, cost))

                    val_inputs, val_targets = mnist.validation.next_batch(batch_size)
                    val_loss = session.run(loss,
                                           feed_dict={inputs: val_inputs, targets: val_targets})

                    print("Worker: {}, Step: {},  Validation Loss: {}".format(FLAGS.task_index, step, val_loss))

                supervisor.stop()

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
