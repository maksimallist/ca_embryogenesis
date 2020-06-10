import matplotlib.pylab as pl
import numpy as np

from embryogenesis.model.old_ca_model import to_rgb
from embryogenesis.model.utils import tile2d, imwrite, imshow


def generate_pool_figures(pool, step_i, save_path):
    tiled_pool = tile2d(to_rgb(pool.x[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]

    imwrite(save_path + '%04d_pool.jpg' % step_i, tiled_pool)


def visualize_batch(x0, x, step_i, save_path):
    vis0 = np.hstack(to_rgb(x0).numpy())
    vis1 = np.hstack(to_rgb(x).numpy())
    vis = np.vstack([vis0, vis1])

    imwrite(save_path + 'batches_%04d.jpg' % step_i, vis)
    print('batch (before/after):')
    imshow(vis)


def plot_loss(loss_log):
    pl.figure(figsize=(10, 4))
    pl.title('Loss history (log10)')
    pl.plot(np.log10(loss_log), '.', alpha=0.1)
    pl.show()
