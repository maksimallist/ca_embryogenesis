import json

import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

from embryogenesis.model.utils import to_rgba


@tf.function
def make_circle_masks(n, h, w):
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = tf.cast(x * x + y * y < 1.0, tf.float32)

    return mask


def loss_f(x, pad_target):
    return tf.reduce_mean(tf.square(to_rgba(x) - pad_target), [-2, -3, -1])


@tf.function
def train_step(ca, trainer, x, pad_target):
    iter_n = tf.random.uniform([], 64, 96, tf.int32)
    with tf.GradientTape() as g:
        for _ in tf.range(iter_n):
            x = ca(x)
        loss = tf.reduce_mean(loss_f(x, pad_target))
    grads = g.gradient(loss, ca.weights)
    grads = [g / (tf.norm(g) + 1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))

    return x, loss


def export_model(ca, base_fn, channel_n: int):
    ca.save_weights(base_fn)
    # todo: fix this error; maybe rewrite save function;
    cf = ca.call.get_concrete_function(x=tf.TensorSpec([None, None, None, channel_n]),
                                       fire_rate=tf.constant(0.5),
                                       angle=tf.constant(0.0),
                                       step_size=tf.constant(1.0))

    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {'format': 'graph-model',
                  'modelTopology': graph_json,
                  'weightsManifest': []}

    with open(base_fn + '.json', 'w') as f:
        json.dump(model_json, f)
