from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from core.image_utils import visualize_batch, generate_pool_figures, to_rgb


class ExperimentWatcher:
    def __init__(self,
                 root: Path,
                 exp_name: str,
                 target: np.array):
        # experiments infrastructure attributes
        self.root = root
        self.exp_name = exp_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.target = target
        self.checkpoints_folder = None
        self.pictures_folder = None
        self.last_pictures_folder = None
        self.tensorboard_logs = None

    def prepare_exp_folder(self):
        exp_root = self.root.joinpath(self.exp_name)
        exp_root.mkdir(parents=True, exist_ok=False)

        self.checkpoints_folder = exp_root.joinpath('checkpoints')
        self.checkpoints_folder.mkdir()

        self.pictures_folder = exp_root.joinpath('pictures')
        self.pictures_folder.mkdir()

        self.tensorboard_logs = exp_root.joinpath('tensorboard_logs')
        self.tensorboard_logs.mkdir()

        file_writer = tf.summary.create_file_writer(str(self.tensorboard_logs))
        file_writer.set_as_default()
        # Using the file writer, log the target image.
        tf.summary.image("Target image", to_rgb(self.target)[None, ...], step=0)

    def create_rule_imprint(self, trainable_rule: Model, train_step: int, pre_state: np.array, post_state: np.array):
        model_path = self.checkpoints_folder.joinpath(str(train_step))
        model_path.mkdir()
        trainable_rule.save(filepath=str(model_path), overwrite=True, save_format="tf")

        self.last_pictures_folder = self.pictures_folder.joinpath(str(train_step))
        self.last_pictures_folder.mkdir()
        visualize_batch(pre_state=pre_state,
                        post_state=post_state,
                        train_step=train_step,
                        save_path=str(self.last_pictures_folder),
                        jupyter=False)

    def log(self, step, loss, petri_dish, trainable_rule, previous_state_batch, next_state_batch):
        if step % 100 == 0:
            self.create_rule_imprint(trainable_rule=trainable_rule,
                                     train_step=step,
                                     pre_state=previous_state_batch,
                                     post_state=next_state_batch)

            tf.summary.scalar('loss_log', data=np.log10(loss), step=step)
            tf.summary.image("Example of CA figures", to_rgb(previous_state_batch[0])[None, ...], step=step)

        if step % 10 == 0:
            pool_states = petri_dish.set_of_petri_dishes
            pool_figures = generate_pool_figures(pool_states=pool_states,
                                                 train_step=step,
                                                 save_path=str(self.last_pictures_folder),
                                                 return_pool=True)

            tf.summary.image("Pool CA figures", pool_figures[None, ...], max_outputs=25, step=step)

        print(f"\r step: {step}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')


class SimpleTFKerasTrainer:
    def __init__(self,
                 data_generator,
                 model: Model,
                 optimizer,
                 loss: Optional[Callable],
                 watcher):
        self.data_generator = data_generator
        self.watcher = watcher
        self.model = model
        self.loss = self.define_loss(loss)

        self.optimizer = optimizer
        # compilation keras model in other scope
        # self.model.compile(optimizer=self.optimizer, loss=self.loss)

    @tf.function
    def ca_loss_function(self, batch_x: np.array, batch_y):
        batch_cells = batch_x[..., :4]  # to rgba
        return tf.reduce_mean(tf.square(batch_cells - batch_y), [-2, -3, -1])

    def define_loss(self, loss: Optional[Callable]):
        if loss:
            return loss
        else:
            return self.ca_loss_function

    def train_step(self, grad_norm_value: float, grow_steps: Tuple[int, int]):
        batch_x, batch_y = self.data_generator.sample()

        iter_n = tf.random.uniform([], grow_steps[0], grow_steps[1], tf.int32)

        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                batch_x = self.model(batch_x)

            loss = tf.reduce_mean(self.loss(batch_x, batch_y))

        grads = g.gradient(loss, self.model.weights)
        grads = [g / (tf.norm(g) + grad_norm_value) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.weights))

        self.data_generator.commit(batch_x)

        return loss, batch_x

    def train(self, train_steps: int, grad_norm_value: float, grow_steps: Tuple[int, int]):
        for step in range(train_steps + 1):
            loss, next_state_batch = self.train_step(grad_norm_value, grow_steps)
            # TODO: надо подумать над форматом API, в рамках которого мы могли бы свободно оперировать дополнительными
            #  объектами, требуемыми нам в рамках обучения
            self.watcher.log(step, loss, self.model, next_state_batch)


class PoolTFKerasTrainer:
    def __init__(self,
                 data_generator,
                 model: Model,
                 optimizer,
                 loss: Optional[Callable],
                 watcher):
        self.data_generator = data_generator
        self.watcher = watcher
        self.model = model
        self.loss = self.define_loss(loss)
        self.optimizer = optimizer

    @tf.function
    def ca_loss_function(self, batch_x: np.array, batch_y):
        batch_cells = batch_x[..., :4]  # to rgba
        return tf.reduce_mean(tf.square(batch_cells - batch_y), [-2, -3, -1])

    def define_loss(self, loss: Optional[Callable]):
        if loss:
            return loss
        else:
            return self.ca_loss_function

    @tf.function
    def make_circle_damage_masks(self, n: int):
        x = tf.linspace(-1.0, 1.0, target_shape[1])[None, None, :]
        y = tf.linspace(-1.0, 1.0, target_shape[0])[None, :, None]

        center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
        r = tf.random.uniform([n, 1, 1], 0.1, 0.4)

        x, y = (x - center[0]) / r, (y - center[1]) / r
        mask = tf.cast(x * x + y * y < 1.0, tf.float32)

        return mask

    # TODO: fix and rewrite
    def train_step(self, grad_norm_value: float, grow_steps: Tuple[int, int]):
        previous_state_batch, cells_idx, targets = self.data_generator.sample()

        loss_rank = self.loss(previous_state_batch, targets).numpy().argsort()[::-1]

        batch = previous_state_batch[loss_rank]
        batch[:1] = seed

        if damage_n:
            damage = 1.0 - self.make_circle_damage_masks(damage_n).numpy()[..., None]
            batch[-damage_n:] *= damage

        iter_n = tf.random.uniform([], grow_steps[0], grow_steps[1], tf.int32)

        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                batch_x = self.model(batch_x)

            loss = tf.reduce_mean(self.loss(batch_x, batch_y))

        grads = g.gradient(loss, self.model.weights)
        grads = [g / (tf.norm(g) + grad_norm_value) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.weights))

        self.data_generator.commit(batch_cells=next_state_batch, cells_idx=cells_idx)

        return loss, next_state_batch

    def train(self, train_steps: int, grad_norm_value: float, grow_steps: Tuple[int, int]):
        for step in range(train_steps + 1):
            loss, next_state_batch = self.train_step(grad_norm_value, grow_steps)
            # TODO: надо подумать над форматом API, в рамках которого мы могли бы свободно оперировать дополнительными
            #  объектами, требуемыми нам в рамках обучения
            self.watcher.log(step, loss, self.model, next_state_batch)
