from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model

from core.image_utils import to_rgb


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
        self.exp_name = exp_name + '_' + datetime.now().strftime("%d.%m.%Y-%H.%M")
        self.target = target
        self.checkpoints_folder = None
        self.pictures_folder = None
        self.last_pictures_folder = None
        self.tensorboard_logs = None
        self.experiments_preparation()

    def create_folders(self):
        exp_root = self.root.joinpath(self.exp_name)
        exp_root.mkdir(parents=True, exist_ok=False)

        self.checkpoints_folder = exp_root.joinpath('checkpoints')
        self.checkpoints_folder.mkdir()

        self.pictures_folder = exp_root.joinpath('pictures')
        self.pictures_folder.mkdir()

        self.tensorboard_logs = exp_root.joinpath('tensorboard_logs')
        self.tensorboard_logs.mkdir()

    def create_summary_writer(self):
        file_writer = tf.summary.create_file_writer(str(self.tensorboard_logs))
        file_writer.set_as_default()

        # Using the file writer, log the target image.
        # todo: move 'to_rgb' function
        target_image = to_rgb(self.target)[None, ...]
        tf.summary.image("Target image", target_image, step=0)

    def experiments_preparation(self):
        self.create_folders()
        self.create_summary_writer()

    def save_model(self, trainable_rule: Model, train_step: int):
        model_path = self.checkpoints_folder.joinpath("train_step_" + str(train_step))
        model_path.mkdir()
        trainable_rule.save(filepath=str(model_path), overwrite=True, save_format="tf")

    # TODO: upgrade
    def save_ca_state_as_image(self, train_step: int, post_state: np.array, max_img_count: int = 5):
        self.last_pictures_folder = self.pictures_folder.joinpath("train_step_" + str(train_step))
        self.last_pictures_folder.mkdir()

        path = open(str(self.last_pictures_folder) + f"/batches_{train_step}.jpeg", 'wb')

        image = np.hstack(to_rgb(post_state)[:max_img_count])
        image = np.uint8(np.clip(image, 0, 1) * 255)
        image = Image.fromarray(image)
        image.save(path, 'jpeg', quality=95)

        tf.summary.image("Example of CA figures", to_rgb(post_state[0])[None, ...], step=train_step)

    # TODO: fix
    # def generate_pool_figures(pool_states: np.array, train_step: int, save_path: str) -> None:
    #     tiled_pool = tile2d(to_rgb(pool_states[:49]))
    #     fade = np.linspace(1.0, 0.0, 72)
    #     ones = np.ones(72)
    #     tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    #     tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    #     tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    #     tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
    #
    #     image_save(save_path + f"/{train_step}_pool.jpg", tiled_pool)

    def log(self, step, loss, trainable_rule, next_state_batch):
        if step % 10 == 0:
            tf.summary.scalar('loss_log', data=np.log10(loss), step=step)

        if step % 100 == 0:
            print(f"\r step: {step}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')

        if step % 1000 == 0:
            self.save_model(trainable_rule, step)
            self.save_ca_state_as_image(step, next_state_batch, max_img_count=5)
            # pool_figures = generate_pool_figures(pool_states=pool_states,
            #                                      train_step=step,
            #                                      save_path=str(self.last_pictures_folder),
            #                                      return_pool=True)


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
