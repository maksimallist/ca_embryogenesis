from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from core.image_utils import visualize_batch, generate_pool_figures, to_rgb


@tf.function  # TODO: вынести перевод в rgba формат во вне тела функции
def l2_loss(batch_x: np.array, batch_y: np.array):
    batch_cells = batch_x[..., :4]  # to rgba
    return tf.reduce_mean(tf.square(batch_cells - batch_y), [-2, -3, -1])


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


class TFKerasTrainer:
    def __init__(self,
                 data_generator,
                 model: Model,
                 optimizer,
                 watcher,
                 loss_function: Optional[Callable]):
        self.data_generator = data_generator
        self.watcher = watcher
        self.model = model  # compiled keras model

        self.optimizer = optimizer
        self.loss_function = loss_function
        # compilation keras model can be in other scope
        # self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train_step(self, grad_norm_value: float, grow_steps: Tuple[int, int]):
        iter_n = tf.random.uniform([], grow_steps[0], grow_steps[1], tf.int32)
        batch_states, targets = self.data_generator.sample()  # batch_states: Tuple[np.array, list indexes]

        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                batch_x = self.model(batch_x)

            loss = tf.reduce_mean(self.loss_function(batch_x, targets))

        grads = g.gradient(loss, self.model.weights)
        grads = [g / (tf.norm(g) + grad_norm_value) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.weights))

        # insert new ca tensors in pool
        self.data_generator.commit(batch_cells=batch_x, cells_idx=batch_states[1])

        return loss, batch_x

    def train(self, train_steps: int, grad_norm_value: float, grow_steps: Tuple[int, int]) -> None:
        for step in range(1, train_steps, 1):
            loss, next_state_batch = self.train_step(grad_norm_value, grow_steps)
            self.watcher.log(step, loss, self.model, next_state_batch)
