from typing import Union

import numpy as np
import tensorflow as tf

from embryogenesis.model.utils import tile2d, imwrite, imshow


@tf.function
def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


def generate_pool_figures(pool_states: np.array,
                          train_step: int,
                          save_path: str,
                          return_pool: bool = False) -> Union[np.array, None]:
    tiled_pool = tile2d(to_rgb(pool_states[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]

    imwrite(save_path + '/%04d_pool.jpg' % train_step, tiled_pool)

    if return_pool:
        return np.asarray(tiled_pool)


def visualize_batch(pre_state: np.array,
                    post_state: np.array,
                    train_step: int,
                    save_path: str,
                    jupyter: bool = False) -> None:
    vis0 = np.hstack(to_rgb(pre_state).numpy())
    vis1 = np.hstack(to_rgb(post_state).numpy())
    vis = np.vstack([vis0, vis1])
    # save pictures
    imwrite(save_path + '/batches_%04d.jpg' % train_step, vis)
    # visualize pictures in notebook or operation system
    if jupyter:
        print('batch (before/after):')
        imshow(vis)
