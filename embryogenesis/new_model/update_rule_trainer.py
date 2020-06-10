from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from embryogenesis.new_model.petri_dish import PetriDish
from embryogenesis.new_model.utils import load_emoji, to_rgba


class UpdateRuleTrainer:
    def __init__(self,
                 root: Path,
                 petri_dish: PetriDish,
                 rule_model: Model,
                 target_image: str,
                 max_image_size: int,
                 batch_size: int,
                 train_steps: int,
                 use_pattern_pool: bool,
                 damage_n: Optional[int] = None,
                 target_padding: int = 16) -> None:

        self.petri_dish = petri_dish
        self.trainable_rule = rule_model

        # target attributes
        target_image = load_emoji(target_image, max_size=max_image_size)
        self.p = target_padding
        self.target = tf.pad(target_image, [(self.p, self.p), (self.p, self.p), (0, 0)])
        self.target_shape = self.target.shape[:2]

        # train attributes
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.use_pattern_pool = use_pattern_pool
        self.damage_n = damage_n

        lr = 2e-3
        lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)

    @tf.function
    def loss_f(self, batch_cells: np.array):
        return tf.reduce_mean(tf.square(to_rgba(batch_cells) - self.target), [-2, -3, -1])

    @tf.function
    def make_circle_masks(self, n: int):
        x = tf.linspace(-1.0, 1.0, self.target_shape[1])[None, None, :]
        y = tf.linspace(-1.0, 1.0, self.target_shape[0])[None, :, None]

        center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
        r = tf.random.uniform([n, 1, 1], 0.1, 0.4)

        x, y = (x - center[0]) / r, (y - center[1]) / r
        mask = tf.cast(x * x + y * y < 1.0, tf.float32)

        return mask

    @tf.function
    def train_step(self, input_tensor):
        # todo: убрать хардкодинг
        iter_n = tf.random.uniform([], 64, 96, tf.int32)  # sample random int from 64 to 96

        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                input_tensor = self.trainable_rule(input_tensor)
            loss = tf.reduce_mean(self.loss_f(input_tensor))

        grads = g.gradient(loss, self.trainable_rule.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.trainable_rule.weights))

        return input_tensor, loss

    def train(self):
        for step_i in range(self.train_steps + 1):
            if self.use_pattern_pool:
                seed = self.petri_dish.make_seed(return_seed=True)
                batch, cells_idx = self.petri_dish.sample(batch_size=self.batch_size)
                loss_rank = self.loss_f(batch).numpy().argsort()[::-1]

                batch = batch[loss_rank]
                batch[:1] = seed

                if self.damage_n:
                    damage = 1.0 - self.make_circle_masks(self.damage_n).numpy()[..., None]
                    batch[-self.damage_n:] *= damage

                x, loss = self.train_step(batch)
                self.petri_dish.commit(batch_cells=x, cells_idx=cells_idx)

            else:
                batch = self.petri_dish.create_petri_dish(return_dish=True, pool_size=self.batch_size)
                x, loss = self.train_step(batch)

            # if step_i % 10 == 0:
            #     generate_pool_figures(pool, step_i)

            if step_i % 100 == 0:
                self.trainable_rule.save(str(self.root))
                # clear_output()
                # visualize_batch(x0, x, step_i)
                # plot_loss(loss_log)

            print(f"\r step: {step_i}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')
