import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
import os
import sys

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def train(sess):
    _X, _y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                                 n_redundant=2, n_classes=2, n_clusters_per_class=10,
                                 random_state=1)
    _y = np.reshape(_y, [_y.shape[0], 1])

    learning_rate = 0.01
    epoch = 1000

    x = tf.placeholder(tf.float32, [None, _X.shape[1]])
    y = tf.placeholder(tf.float32, [None, 1])

    w = tf.Variable(tf.zeros([_X.shape[1], 1]), name='w')
    b = tf.Variable(tf.zeros([1]), name='b')

    logits = tf.matmul(x, w) + b
    pred = tf.sigmoid(logits)

    l = tf.constant(0.1)
    l2_norm = l * tf.nn.l2_loss(w)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)) + l2_norm
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    sess.run(tf.global_variables_initializer())
    sess.run(optimizer, feed_dict={x: _X, y: _y})

    return (x, pred)

def export(sess, x, scores):
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_scores = tf.saved_model.utils.build_tensor_info(scores)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.PREDICT_INPUTS:
                    tensor_info_x
            },
            outputs={
                tf.saved_model.signature_constants.PREDICT_OUTPUTS:
                    tensor_info_scores
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict': prediction_signature
        },
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op'))
    builder.save(as_text=False)

def main(_):
    sess = tf.InteractiveSession()
    x, scores = train(sess)
    export(sess, x, scores)

if __name__ == '__main__':
  tf.app.run()
