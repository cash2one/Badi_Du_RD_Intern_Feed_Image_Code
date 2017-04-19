#!/usr/bin/env python
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
'''
File: distributed_train.py
Author: feedrd(feedrd@baidu.com)
Date: 2017/04/16 17:09:39
'''
import tensorflow as tf
import input_set
import resnet_vision
from deployment import model_deploy
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

########Cluster Set###########
tf.app.flags.DEFINE_string('job_name', 'worker', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222', empty for local""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer('task', 0,
                            'Task id of the replica running the training.')
tf.app.flags.DEFINE_bool('sync_replicas', Flase,
                         'Whether or not to synchronize the replicas during training.')
tf.app.flags.DEFINE_integer('replicas_to_aggregate', 1,
                           'The Number of gradients to collect before updating params.')

########Moving Average Set ###

tf.app.flags.DEFINE_float('moving_average_decay', 0.9,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')

########Dataset ##############
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')

########Iteration Set#########
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch_size')
tf.app.flags.DEFINE_integer('max_steps', 50, 'Number of batches to run')

########Learning Set##########
tf.app.flags.DEFINE_float('lrn_rate',0.001,'lrn_rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

########Model Set############
tf.app.flags.DEFINE_float('weight_decay', 0.0004,
                          """regularization""")
tf.app.flags.DEFINE_integer('logits', 512,'Number of out vector')


########Save Set#############
tf.app.flags.DEFINE_string('train_dir', './model',
                           'Directory to keep training outputs.')

tf.app.flags.DEFINE_string('log_root', './model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('trace_every_n_steps', None,
                            'The frequency with which logs are trace.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 600,
                            'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer('save_interval_secs', 600,
                            'The frequency with which the model is saved, in seconds.')


#########Print Set###########
tf.app.flags.DEFINE_integer('log_every_n_steps', 10,
                            'The frequency with which logs are print.')

#########Fine-Tuning#########
tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
                           'Comma-separated list of scopes of variables to exclude when restoring '
                           'from a checkpoint.')
tf.app.flags.DEFINE_string('trainable_scopes', None,
                           'Comma-separated list of scopes to filter the set of variables to train.'
                           'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', False,
                            'When restoring a checkpoint would ignore missing variables.')

Num_examples_per_epoch = 120000


def _get_init_fn():
    if FLAGS.checkpoint_path is None:
        return None
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                       for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break

        if not excluded:
            variables_to_restore.append(var)
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
                                    checkpoint_path,
                                    variables_to_restore,
                                    ignore_missing_vars=FLAGS.ignore_missing_vars)



def _get_variables_to_train():
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    return variables_to_train




def main(_):
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    ps_hosts = [] if len(FLAGS.ps_hosts) < 1 else FLAGS.ps_hosts.split(',')
    worker_hosts = [] if len(FLAGS.worker_hosts) < 1 else FLAGS.worker_hosts.split(',')
    num_ps_tasks = len(ps_hosts)
    num_replicas = 1 if len(worker_hosts) < 1 else len(worker_hosts)
    target = FLAGS.master
    if len(ps_hosts) > 0:
        tf.logging.info('PS hosts are: %s' % ps_hosts)
        tf.logging.info('Worker hosts are: %s' % worker_hosts)
        cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                            'worker': worker_hosts})
        server = tf.train.Server(
                                 {'ps': ps_hosts,
                                 'worker': worker_hosts},
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task)

        target = server.target
        if FLAGS.job_name == 'ps':
            server.join()
            return
        
    if FLAGS.task == 0:
        if not tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
                                         num_clones=FLAGS.num_clones,
                                         clone_on_cpu=FLAGS.clone_on_cpu,
                                         replica_id=FLAGS.task,
                                         num_replicas=num_replicas,
                                         num_ps_tasks=num_ps_tasks)

        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        with tf.device(deploy_config.inputs_device()):
            batch_examples, batch_labels = input_set.get_input(FLAGS.train_data_path, FLAGS.batch_size)
            mini_batch_examples = tf.split(batch_examples, num_or_size_splits=FLAGS.num_clones, axis=0)
            mini_batch_labels = tf.split(batch_labels, num_or_size_splits=FLAGS.num_clones, axis=0)


        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        clones = []
        reuse_variables = None
        for i in range(0, FLAGS.num_clones):
            clone_device = deploy_config.clone_device(i)
            with tf.name_scope(deploy_config.clone_scope(i)) as clone_scope:
                with tf.device(clone_device):
                    with tf.variable_scope(tf.get_variable_scope(),
                                                reuse=True if i > 0 else None):
                        outputs = resnet_vision.inference(mini_batch_examples[i], mini_batch_labels[i], FLAGS.weight_decay,
                                                             feature_num=FLAGS.logits, is_training=True, reuse_variables=reuse_variables)
                        reuse_variables = True
                    clones.append(model_deploy.Clone(outputs, clone_scope, clone_device))

        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

            
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                                        FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

            
        with tf.device(deploy_config.optimizer_device()):
            num_batches_per_epoch = (Num_examples_per_epoch /
                                         FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
            lr = tf.train.exponential_decay(FLAGS.lrn_rate,
                                                global_step,
                                                decay_steps,
                                                FLAGS.learning_rate_decay_factor,
                                                staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, 0.9)
            summaries.add(tf.summary.scalar('learning_rate', lr))
            if FLAGS.sync_replicas:
                optimizer = tf.train.SyncReplicasOptimizer(
                                         opt=optimizer,
                                         replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                                         variable_averages=variable_averages,
                                         variables_to_average=moving_average_variables,
                                         total_num_replicas=num_replicas)
            elif FLAGS.moving_average_decay:
                    update_ops.append(variable_averages.apply(moving_average_variables))
                
            variables_to_train = _get_variables_to_train()
            total_loss, clones_gradients = model_deploy.optimize_clones(
                                     clones,
                                     optimizer,
                                     var_list=variables_to_train)
                
            summaries.add(tf.summary.scalar('total_loss', total_loss))
            grad_updates = optimizer.apply_gradients(clones_gradients,
                                                         global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)
            train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                    name='train_op')
            summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                   first_clone_scope))
            summary_op = tf.summary.merge(list(summaries), name='summary_op')
                
            gpu_opts = {}
            gpu_opts["allow_growth"] = True
            gpu_options = tf.GPUOptions(**gpu_opts)
            device_filters = []
            if len(ps_hosts) > 0:
                device_filters = ["/job:ps", "/job:worker/task:%d" % FLAGS.task]

            slim.learning.train(
                        train_tensor,
                        logdir=FLAGS.train_dir,
                        global_step=global_step,
                        master=target,
                        is_chief=(FLAGS.task == 0),
                        init_fn=_get_init_fn(),
                        summary_op=summary_op,
                        number_of_steps=FLAGS.max_steps,
                        log_every_n_steps=FLAGS.log_every_n_steps,
                        trace_every_n_steps=FLAGS.trace_every_n_steps,
                        save_summaries_secs=FLAGS.save_summaries_secs,
                        save_interval_secs=FLAGS.save_interval_secs,
                        sync_optimizer=optimizer if FLAGS.sync_replicas else None,
                        session_config=tf.ConfigProto(
                                        allow_soft_placement=True,
                                        gpu_options=gpu_options,
                                        log_device_placement=FLAGS.log_device_placement,
                                        device_filters=device_filters))






if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

