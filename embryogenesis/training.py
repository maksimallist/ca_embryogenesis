import json

import matplotlib.pylab as pl
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

from embryogenesis.ca_model import to_rgb
from embryogenesis.utils import tile2d, imwrite, imshow, to_rgba


class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


@tf.function
def make_circle_masks(n, h, w):
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = tf.cast(x * x + y * y < 1.0, tf.float32)

    return mask


def export_model(ca, base_fn, channel_n: int):
    ca.save_weights(base_fn)

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


def generate_pool_figures(pool, step_i):
    tiled_pool = tile2d(to_rgb(pool.x[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]

    imwrite('train_log/%04d_pool.jpg' % step_i, tiled_pool)


def visualize_batch(x0, x, step_i):
    vis0 = np.hstack(to_rgb(x0).numpy())
    vis1 = np.hstack(to_rgb(x).numpy())
    vis = np.vstack([vis0, vis1])

    imwrite('train_log/batches_%04d.jpg' % step_i, vis)
    print('batch (before/after):')
    imshow(vis)


def plot_loss(loss_log):
    pl.figure(figsize=(10, 4))
    pl.title('Loss history (log10)')
    pl.plot(np.log10(loss_log), '.', alpha=0.1)
    pl.show()


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

# for i in range(8000 + 1):
#     if USE_PATTERN_POOL:
#         batch = pool.sample(BATCH_SIZE)
#         x0 = batch.x
#         loss_rank = loss_f(x0).numpy().argsort()[::-1]
#         x0 = x0[loss_rank]
#         x0[:1] = seed
#         if DAMAGE_N:
#             damage = 1.0 - make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
#             x0[-DAMAGE_N:] *= damage
#     else:
#         x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
#
#     x, loss = train_step(x0)
#
#     if USE_PATTERN_POOL:
#         batch.x[:] = x
#         batch.commit()
#
#     step_i = len(loss_log)
#     loss_log.append(loss.numpy())
#
#     if step_i % 10 == 0:
#         generate_pool_figures(pool, step_i)
#     if step_i % 100 == 0:
#         clear_output()
#         visualize_batch(x0, x, step_i)
#         plot_loss(loss_log)
#         export_model(ca, 'train_log/%04d' % step_i)
#
#     print('\r step: %d, log10(loss): %.3f' % (len(loss_log), np.log10(loss)), end='')
