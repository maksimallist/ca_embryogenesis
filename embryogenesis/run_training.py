import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from embryogenesis.model.ca_model import CAModel
from embryogenesis.model.train import loss_f, make_circle_masks, train_step, generate_pool_figures, plot_loss, \
    export_model, visualize_batch, SamplePool
from embryogenesis.model.utils import load_emoji

experiment_config = 'train_config.json'
with open(experiment_config, 'r') as conf:
    config = json.load(conf)

target_img = load_emoji("🦎", max_size=48)
# запринтить пример картинки
# imshow(zoom(to_rgb(target_img), 2), fmt='png')
# ----------------------------------------------------------------------------------------------------------------------
p = config['ca_params']['TARGET_PADDING']
pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]

seed = np.zeros(shape=[h, w, config['ca_params']['CHANNEL_N']], dtype=np.float32)
seed[h // 2, w // 2, 3:] = 1.0

ca = CAModel(channel_n=config['ca_params']['CHANNEL_N'], fire_rate=config['ca_params']['CELL_FIRE_RATE'])

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

loss0 = loss_f(seed, pad_target=pad_target).numpy()
pool = SamplePool(x=np.repeat(seed[None, ...], config['ca_params']['POOL_SIZE'], 0))
root = Path("/Users/a17264288/PycharmProjects/cellar_automata_experiments/embryogenesis/experiments/exp_test/train_log")
root.mkdir(parents=True, exist_ok=False)
# !mkdir -p train_log && rm -f train_log/*

loss_log = []
exp_map = config['experiment_config']['EXPERIMENT_MAP']
EXPERIMENT_N = exp_map[config['experiment_config']['EXPERIMENT_TYPE']]
use_pattern_pool = [0, 1, 1][EXPERIMENT_N]
damage_n = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch
batch_size = config['ca_params']['BATCH_SIZE']
channel_n = config['ca_params']['CHANNEL_N']

for i in range(8000 + 1):
    if use_pattern_pool:
        batch = pool.sample(batch_size)
        x0 = batch.x
        loss_rank = loss_f(x0, pad_target=pad_target).numpy().argsort()[::-1]
        x0 = x0[loss_rank]
        x0[:1] = seed
        if damage_n:
            damage = 1.0 - make_circle_masks(damage_n, h, w).numpy()[..., None]
            x0[-damage_n:] *= damage
    else:
        x0 = np.repeat(seed[None, ...], batch_size, 0)

    x, loss = train_step(ca=ca, trainer=trainer, x=x0, pad_target=pad_target)  # ca, trainer, x, pad_target

    if use_pattern_pool:
        batch.x[:] = x
        batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    if step_i % 10 == 0:
        generate_pool_figures(pool, step_i, str(root))
    if step_i % 100 == 0:
        from IPython.display import clear_output

        clear_output()

        visualize_batch(x0, x, step_i)
        plot_loss(loss_log)
        export_model(ca, 'train_log/%04d' % step_i, channel_n=channel_n)

    print('\r step: %d, log10(loss): %.3f' % (len(loss_log), np.log10(loss)), end='')
