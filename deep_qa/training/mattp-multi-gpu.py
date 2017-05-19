
import os
import tensorflow as tf
import numpy as np
import time

from pprint import pprint
from keras import backend as K

from .train_utils import _clip_grads, _average_gradients
from .train_utils import _scale_grads, slice_batch

# notes on keras model
#
# Tensorflow ops to train:
#   loss_op = model.total_loss
#   Placeholders for feed_dict:
#       X tensors: model._feed_inputs with names model._feed_input_names
#       y tensors: feed_y = model._feed_targets with names model.output_names
#       sample weight tensors: model._feed_sample_weights
#       [0 - 1] for K.learning_phase(), only if model.uses_learning_phase
#
# notes on tf variable_scope vs name_scope:
#   they are the same thing, but for variables and ops!  Opening
#   tf.variable_scope implicity opens name_scope, but not the other way


class MultiGPUModel(object):

    def __init__(self, n_gpus: int, builders, builder_options,
                 use_moving_averages=None):
        '''
        builders = {
            'model': model_builder,
            'optimizer': optimizer_builder,
            'summary': summary_builder
        }
        builder_options = passed into each of the builders

        Each builder interface is:
            model_builder(options) returns a keras model instance
            optimizer_builder(global_step, options) returns a tensorflow
              optimizer instance
            summary_builder(options, model[0]) returns a tuple of
                tensorflow scalar/histogram summary ops
                (scalar_summaries, hist_summaries)
        use_moving_averages: if not None, then specifies the decay
            parameter for moving averages of variables for test (typical
            values are 0.999 or 0.99)
        '''
        #  builders = {'model': model_builder, 'optimizer': ..., 'summary': ..}
        self._builders = builders
        self._builder_options = builder_options
        self._n_gpus = n_gpus
        self._use_moving_averages = use_moving_averages
        self._build_graph(builder_options, n_gpus)
        self._gpu_func = None
        self._summary_writer = None

    @property
    def builder_options(self):
        return self._builder_options

    def restore_from_checkpoint(self, ckpt_file, test=False):
        sess = K.get_session()
        loader = tf.train.Saver(self._get_saver_spec(test=test, save=False))
        loader.restore(sess, ckpt_file)

    def _build_graph(self, options, n_gpus):
        with tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # set up the optimizer
            self.opt = self._builders['optimizer'](global_step, options)

            # calculate the gradients on each GPU
            tower_grads = []
            models = []
            train_loss = tf.get_variable(
                'train_loss', [],
                initializer=tf.constant_initializer(0.0), trainable=False)
            with tf.variable_scope('t') as scope:
                for k in range(n_gpus):
                    with tf.device('/gpu:%d' % k), tf.name_scope('gpu_%d' % k):
                        # calculate the loss for one model replica
                        model = self._builders['model'](options)
                        loss = model.total_loss
                        models.append(model)
                        # reuse variables
                        scope.reuse_variables()
                        # get gradients
                        grads = self.opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        # keep track of loss across all GPUs
                        train_loss += loss

            # calculate the mean of each gradient across all GPUs
            grads = _average_gradients(tower_grads)
            # now clip the gradients
            clip_norm_value = options.get('clip_norm_value', None)
            grads, norm_summaries = _clip_grads(
                grads, clip_norm_value, global_step)
            # scale the gradients if needed
            grad_scale_spec = options.get('grad_scale_spec', None)
            if grad_scale_spec is not None:
                grads = _scale_grads(grads, grad_scale_spec)
            # apply the gradients to create the training operation
            apply_grad_op = self.opt.apply_gradients(
                grads, global_step=global_step)

            # create the training op
            if self._use_moving_averages:
                self.ema = tf.train.ExponentialMovingAverage(
                    decay=self._use_moving_averages)
                # get an op to update moving averages of trainable vars

                # CAUTION!! This is an extremely subtle problem.
                # the EMA in tensorflow RE-RUNS the variable initialization
                # op!  This causes all kinds of havoc with the model, esp.
                # for variables that were set to sensible initial values
                # AFTER initialization (LIKE bias of LSTM).
                # SO: we need to stash away values, create moving averages,
                # then set them again.
                trainable_vars = tf.trainable_variables()
                init_vals = {
                    v.name: (v, K.get_value(v)) for v in trainable_vars}
                maintain_averages_op = self.ema.apply(trainable_vars)
                with tf.control_dependencies([apply_grad_op]):
                    train_op = tf.group(maintain_averages_op)
                # now reset the initial values
                for v_name, (v, val) in init_vals.items():
                    K.set_value(v, val)
                for ema_v in tf.global_variables():
                    if 'ExponentialMovingAverage' not in ema_v.name:
                        continue
                    ema_v_name = ema_v.name
                    v_name = ema_v_name.replace('/ExponentialMovingAverage', '')
                    val = init_vals[v_name][1]
                    K.set_value(ema_v, val)
                del init_vals
            else:
                train_op = apply_grad_op
                self.ema = None

            # NOW SOME SUMMARIES  ----------------------
            train_summary = tf.summary.scalar(
                'train_loss', train_loss / n_gpus)


            summary_ops = [train_summary]
            # any metrics that keras has collected
            merged_metrics = []
            if models[0].metrics is not None:
                # merge the metrics across GPUs
                for k in range(len(models[0].metrics)):
                    mname = models[0].metrics[0]
                    mtensor = tf.reduce_mean([mm.metrics_tensors[k]
                                              for mm in models])
                    summary_ops.append(tf.summary.scalar(mname, mtensor))
                    merged_metrics.append(mtensor)


            # summary ops specified in builder
            summary_builder = self._builders['summary']
            if summary_builder is not None:
                _so, _hs = summary_builder(options, models[0])
                summary_ops.extend(_so)

            summary_op = tf.summary.merge(summary_ops + norm_summaries)

        self.ops = {
            'train_op': train_op,
            'summary_op': summary_op,
            'train_loss': train_loss,
            'merged_metrics_op': merged_metrics,
        }
        self.variables = {
            'global_step': global_step,
        }

        self.models = models

        self.uses_learning_phase = self.models[0].uses_learning_phase

    def summary(self):
        print("ALL VARIABLES and shape")
        info = [(v.name, K.int_shape(v)) for v in tf.global_variables()]
        pprint(info)

        self.models[0].summary()

    def _make_functions(self):
        if self._gpu_func is None:
            inputs = []
            updates = []
            for model in self.models:
                model_inputs = (model._feed_inputs + model._feed_targets +
                                model._feed_sample_weights)
                inputs.extend(model_inputs)
                updates.extend(model.updates)
            if self.models[0].uses_learning_phase and \
                    not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]

            outputs = [self.ops['train_op'], self.ops['train_loss'],
                       self.ops['summary_op']] + self.ops['merged_metrics_op']

            # Gets loss and metrics. Updates weights at each call.
            # _gpu_func is a callable that takes a list of numpy arrays
            # corresponding to inputs and returns the output arrays, while
            # running the updates
            self._gpu_func = {}
            self._gpu_func['train'] = K.function(inputs, outputs,
                                                 updates=updates)

            # a new function that also runs histograms summaries
            hist_outputs = list(outputs)
            hist_outputs.append(self.ops['hist_summary_op'])
            self._gpu_func['train_hist'] = K.function(inputs, hist_outputs,
                                                      updates=updates)

            # a function that does testing
            metrics_tensors = self.ops['merged_metrics_op']
            with tf.variable_scope('test_loop'):
                n_agg = 2 + len(metrics_tensors)
                test_agg = tf.get_variable('test_agg',
                                           initializer=tf.zeros_initializer(),
                                           shape=(n_agg,))
                tf.get_variable_scope().reuse_variables()

            # need to run this once when starting test evaluation
            start_test_op = tf.assign(test_agg, tf.zeros(tf.shape(test_agg)))
            test_agg_update_op = tf.assign_add(
                test_agg,
                tf.stack([1.0, self.ops['train_loss']] +
                         metrics_tensors))
            # the summary op to run once, after finished testing
            test_summaries = [
                tf.summary.scalar('test_loss', test_agg[1] / test_agg[0])]
            if self.models[0].metrics is not None:
                for k, mname in enumerate(self.models[0].metrics, start=2):
                    test_summaries.append(
                        tf.summary.scalar('test_' + mname,
                                          test_agg[k] / test_agg[0]))
            test_summary_op = tf.summary.merge(test_summaries)
            self.ops['test_summary_op'] = test_summary_op
            self.ops['test_start_op'] = start_test_op
            self.ops['test_agg'] = test_agg

            test_outputs = [self.ops['train_loss'], test_agg_update_op] + \
                           metrics_tensors
            self._gpu_func['test'] = K.function(inputs, test_outputs,
                                                updates=updates)

    def prepare_inputs(self, batch, train=True):
        # slice X and y
        X, y = batch
        X_sliced = slice_batch(X, self._n_gpus)
        y_sliced = slice_batch(y, self._n_gpus)
        batch_size = int(X[list(X_sliced.keys())[0]].shape[0] / self._n_gpus)
        sample_weights = np.ones(batch_size, )

        inps = []
        for k, model in enumerate(self.models):
            for name in model._feed_input_names:
                inps.append(X_sliced[name][k])
            for name in model.output_names:
                inps.append(y_sliced[name][k])
            # now the sample weights, one per output tensor
            for _ in model.output_names:
                inps.append(sample_weights)
        # finally learning phase
        if self.uses_learning_phase:
            if train:
                inps.append(1.)
            else:
                inps.append(0.)

        return inps

    def test_parallel(self, test_data, batch_no=None):
        # test_data is a generator
        self._make_functions()
        sess = K.get_session()

        # initialize the accumulators
        ret = sess.run([self.ops['test_start_op']])

        # compute metrics
        for batch in test_data:
            # slice the input in the batch for the feed_dict
            inputs = self.prepare_inputs(batch, False)
            ret = self._gpu_func['test'](inputs)

        # summaries
        ret = sess.run([self.ops['test_summary_op']])
        # write to tensorboard
        if self._summary_writer and batch_no:
            self._summary_writer.add_summary(ret[0], batch_no)

        # get the return values
        test_agg = sess.run([self.ops['test_agg']])[0]
        keys = ['loss']
        if self.models[0].metrics:
            keys += self.models[0].metrics

        ret = {}
        for i, key in enumerate(keys, start=1):
            ret[key] = test_agg[i] / test_agg[0]

        return ret

    def _get_saver_spec(self, save=True, test=False):
        # get something we can pass into tf.train.Saver constructor
        # that handles the use_moving_averages flag
        # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        if not self._use_moving_averages:
            ret = tf.global_variables()
        else:
            # for saving and loading for train, we just save everything
            if save or not test:
                ret = tf.global_variables()
            else:
                # for testing, we load the EMA variables in for the actual
                # variables
                ret = self.ema.variables_to_restore()
        return ret

    def fit_parallel(self, train_data, tf_log_dir=None,
                     dev_data_getter=None, dev_data_func=None,
                     n_epochs=None, n_batches_per_epoch=None,
                     n_examples_per_epoch=None,
                     tf_save_options=None,
                     lr_anneal_fac=None):
        '''
        train_data = generator of n_gpus * batch_size X, y training data
        Two ways to track dev results each epoch:
            1.  Pass dev_data_getter = a callable that returns a generator of
                n_gpus * batch_size X, y dev data
                If using this method, then the returned generator is used
                with self.test_parallel
            2.  Pass dev_data_func = (callable(multi_model), [metric_keys])
                They first element is a callable that accepts the multi_model
                as input and returns a dictionary with dev results.
                The second element is a list of the expected return keys
                from the callable.
        tf_log_dir = write tensorboard logs here
        tf_save_options = a dict with options for saving weights
            {'tf_save_dir': save checkpoint here,
             'dev_results_key': a string key in dev results to only
                    save best results}
        if lr_anneal_fac is not None then it is
            the factor to reduce lr every epoch that the dev metric
            decreases.  In this case you must also provide either
            dev_data_getter or dev_data_func
            lr_anneal_fac is either a float, or [fac, threshold].
        '''
        if tf_save_options:
            saver = tf.train.Saver(self._get_saver_spec(save=True),
                                   max_to_keep=2)

        self._make_functions()

        sess = K.get_session()

        if tf_log_dir:
            summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)
            self._summary_writer = summary_writer

        histogram_interval = self._builder_options['summaries'][
            'histogram_interval']
        scalar_interval = self._builder_options['summaries'][
            'scalar_interval']

        # can't specify both a dev data generator and function
        assert not (dev_data_getter and dev_data_func)
        best_dev_results = None
        last_dev_results = None

        # set up the summary ops for dev results if needed
        if dev_data_func:
            dev_keys = dev_data_func[1]
            dev_keys_placeholders = [
                tf.placeholder(tf.float32, shape=())
                for _ in dev_keys]
            so = [tf.summary.scalar('test_' + k, v)
                  for k, v in zip(dev_keys, dev_keys_placeholders)]
            dev_summary = tf.summary.merge(so)

        t1 = time.time()
        if n_batches_per_epoch and n_examples_per_epoch:
            raise ValueError("only specify one of n_batches_per_epoch or "
                             "n_examples_per_epoch but not both")

        if n_batches_per_epoch:
            total_batches = n_epochs * n_batches_per_epoch
        else:
            # TODO: fix this, so that we'll stop training after n_epochs
            total_batches = None

        epoch_examples = 0
        for batch_no, batch in enumerate(train_data, start=1):

            # slice the input in the batch for the feed_dict
            inputs = self.prepare_inputs(batch, True)
            epoch_examples += (inputs[0].shape[0] * self._n_gpus)

            if batch_no % histogram_interval == 0:
                # also run the histogram summaries
                ret = self._gpu_func['train_hist'](inputs)
            else:
                # just run the train_op, summaries
                ret = self._gpu_func['train'](inputs)

            if batch_no % histogram_interval == 0:
                if tf_log_dir:
                    summary_writer.add_summary(ret[-1], batch_no)
            if batch_no % scalar_interval == 0:
                # write the summaries to tensorboard
                if tf_log_dir:
                    summary_writer.add_summary(ret[2], batch_no)
                print("Batch %s, train_loss=%s" % (batch_no, ret[1]))
                print("Total time: %s" % (time.time() - t1))

            # at end of epoch, evaluate on val data, serialize data
            if n_batches_per_epoch:
                end_epoch = batch_no % n_batches_per_epoch == 0
            else:
                end_epoch = epoch_examples >= n_examples_per_epoch

            if end_epoch:
                epoch_examples = 0

                if dev_data_getter:
                    dev_results = self.test_parallel(
                        dev_data_getter(), batch_no)
                elif dev_data_func:
                    dev_results = dev_data_func[0](self)
                    # log results to tensorboard
                    feed_dict = {v: dev_results[k]
                                 for k, v in zip(dev_keys, dev_keys_placeholders)}
                    r = sess.run(dev_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(r, batch_no)
                else:
                    dev_results = None

                # decide whether to save model
                anneal_lr = False
                save_model = False
                if tf_save_options:
                    key = tf_save_options.get('dev_results_key')
                    if key:
                        if best_dev_results is None:
                            # first epoch
                            save_model = True
                            best_dev_results = dev_results
                        elif dev_results[key] > best_dev_results[key]:
                            save_model = True
                            best_dev_results = dev_results
                        else:
                            save_model = False

                        if last_dev_results is not None and (
                                    dev_results[key] < last_dev_results[key]):
                            anneal_lr = True

                    else:
                        # always save
                        save_model = True

                if save_model:
                    print("SAVING MODEL")
                    tf_save_dir = tf_save_options['tf_save_dir']
                    checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=self.variables['global_step'])

                if anneal_lr and lr_anneal_fac:
                    if isinstance(lr_anneal_fac, list):
                        fac, thres = lr_anneal_fac
                    else:
                        fac = lr_anneal_fac
                        thres = -1

                    delta = last_dev_results[key] - dev_results[key]
                    if delta >= thres:
                        print("ANNEALING LEARNING RATE!")
                        lr = self.opt._lr
                        old_lr = K.get_value(lr)
                        new_lr = old_lr * fac
                        K.set_value(lr, new_lr)
                        print("UPDATED LEARNING RATE FROM {0} to {1}".format(
                            old_lr, new_lr))

                last_dev_results = dev_results

            if batch_no == total_batches:
                # done training!
                break
