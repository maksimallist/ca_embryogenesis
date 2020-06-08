import numpy as np
import tensorflow as tf

from embryogenesis.model.train import loss_f, make_circle_masks, train_step, generate_pool_figures, plot_loss, export_model, \
    visualize_batch


def run_training(ca, pool, target_img, target_padding, batch_size, use_pattern_pool, damage_n, loss_log, channel_n):
    # todo: вынести это
    p = target_padding
    pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
    h, w = pad_target.shape[:2]
    seed = np.zeros([h, w, channel_n], np.float32)
    seed[h // 2, w // 2, 3:] = 1.0

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

        x, loss = train_step(x0)

        if use_pattern_pool:
            batch.x[:] = x
            batch.commit()

        step_i = len(loss_log)
        loss_log.append(loss.numpy())

        if step_i % 10 == 0:
            generate_pool_figures(pool, step_i)
        if step_i % 100 == 0:
            from IPython.display import clear_output
            clear_output()

            visualize_batch(x0, x, step_i)
            plot_loss(loss_log)
            export_model(ca, 'train_log/%04d' % step_i, channel_n=channel_n)

        print('\r step: %d, log10(loss): %.3f' % (len(loss_log), np.log10(loss)), end='')
