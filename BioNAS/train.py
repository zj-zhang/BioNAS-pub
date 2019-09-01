import tensorflow as tf
import keras.backend as K
from BioNAS.examples.simple_conv1d_state_space import train_simple_controller ## this can be changed to other scripts

tf.flags.DEFINE_float("gpu_memory_fraction", 0.95, "GPU memory fraction to use.")
tf.flags.DEFINE_string("ps_hosts", "localhost:2222", "Parameter server")
tf.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2225", "Worker server")
tf.flags.DEFINE_string("job_name", 'worker', "'worker' or 'ps'")
tf.flags.DEFINE_integer("task_index", 0, 'Task index')

FLAGS = tf.flags.FLAGS

def train(cluster):

    ## start servers
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        num_tasks = len(cluster.as_dict()['ps'])
        ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(num_tasks, tf.contrib.training.byte_size_load_fn)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/replica:0/task:%d/GPU:0" % FLAGS.task_index,
                cluster=cluster, ps_strategy=ps_strategy)):
            
            sess = tf.Session(server.target)
            K.set_session(sess) 
            train_simple_controller(FLAGS.task_index == 0) ## can be adapted to other scripts

def main(_):
    ## parse flags
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    ## create cluster

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    train(cluster)

if __name__ == '__main__':
    tf.app.run()

