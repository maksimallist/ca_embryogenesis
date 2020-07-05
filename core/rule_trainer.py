from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from tensorflow.keras import Model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from core.image_utils import visualize_batch, generate_pool_figures, to_rgb
from core.petri_dish import PetriDish


class UpdateRuleTrainer:
    def __init__(self,
                 root: Path,
                 exp_name: str,
                 petri_dish: PetriDish,
                 rule_model: Model,
                 target_image: np.array,
                 batch_size: int,
                 train_steps: int,
                 use_pattern_pool: bool,
                 train_ca_step_range: Tuple[int, int],
                 damage_n: Optional[int] = None,
                 learning_rate: float = 2e-3,
                 boundaries: int = 2000,
                 lr_multiplier: float = 0.1,
                 grad_norm_value: float = 1e-8,
                 jupyter: bool = False):
        # main attributes
        self.petri_dish = petri_dish
        self.trainable_rule = rule_model

        # target attributes
        self.target = target_image
        self.target_shape = self.target.shape[:2]

        # train attributes
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.use_pattern_pool = use_pattern_pool
        self.damage_n = damage_n

        self.lr = learning_rate
        self.boundaries = boundaries
        self.lr_multiplier = lr_multiplier
        lr_scheduler = PiecewiseConstantDecay(boundaries=[self.boundaries],
                                              values=[self.lr, self.lr * self.lr_multiplier])
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        self.trainable_rule.compile(optimizer=self.optimizer)  # , loss=self.loss_f

        self.train_ca_step_range = train_ca_step_range
        self.grad_norm_value = grad_norm_value

        # experiments infrastructure attributes
        self.root = root
        self.exp_name = exp_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoints_folder = None
        self.pictures_folder = None
        self.last_pictures_folder = None
        self.tensorboard_logs = None
        self.jupyter = jupyter
        self.prepare_exp_folder()

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

    @tf.function
    def loss_f(self, batch_cells: np.array):
        batch_cells = batch_cells[..., :4]  # to rgba
        return tf.reduce_mean(tf.square(batch_cells - self.target), [-2, -3, -1])

    @tf.function
    def make_circle_damage_masks(self, n: int):
        x = tf.linspace(-1.0, 1.0, self.target_shape[1])[None, None, :]
        y = tf.linspace(-1.0, 1.0, self.target_shape[0])[None, :, None]

        center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
        r = tf.random.uniform([n, 1, 1], 0.1, 0.4)

        x, y = (x - center[0]) / r, (y - center[1]) / r
        mask = tf.cast(x * x + y * y < 1.0, tf.float32)

        return mask

    @tf.function
    def train_step(self, input_tensor):
        # sample random int from train steps range
        iter_n = tf.random.uniform([], self.train_ca_step_range[0], self.train_ca_step_range[1], tf.int32)
        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                input_tensor = self.trainable_rule(input_tensor)

            loss = tf.reduce_mean(self.loss_f(input_tensor))

        grads = g.gradient(loss, self.trainable_rule.weights)
        grads = [g / (tf.norm(g) + self.grad_norm_value) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_rule.weights))

        return input_tensor, loss

    def create_rule_imprint(self, train_step: int, pre_state: np.array, post_state: np.array):
        model_path = self.checkpoints_folder.joinpath(str(train_step))
        model_path.mkdir()
        self.trainable_rule.save(filepath=str(model_path), overwrite=True, save_format="tf")

        self.last_pictures_folder = self.pictures_folder.joinpath(str(train_step))
        self.last_pictures_folder.mkdir()
        visualize_batch(pre_state=pre_state, post_state=post_state, train_step=train_step,
                        save_path=str(self.last_pictures_folder), jupyter=self.jupyter)

    def train_in_pool_mode(self):
        seed = self.petri_dish.make_seed(return_seed=True)
        previous_state_batch, cells_idx = self.petri_dish.sample(batch_size=self.batch_size)

        loss_rank = self.loss_f(previous_state_batch).numpy().argsort()[::-1]

        batch = previous_state_batch[loss_rank]
        batch[:1] = seed

        if self.damage_n:
            damage = 1.0 - self.make_circle_damage_masks(self.damage_n).numpy()[..., None]
            batch[-self.damage_n:] *= damage

        next_state_batch, loss = self.train_step(batch)
        self.petri_dish.commit(batch_cells=next_state_batch, cells_idx=cells_idx)

        return next_state_batch, loss, previous_state_batch

    def train(self):
        for step in range(self.train_steps + 1):
            if self.use_pattern_pool:
                next_state_batch, loss, previous_state_batch = self.train_in_pool_mode()
            else:
                previous_state_batch = self.petri_dish.create_petri_dishes(return_dish=True, pool_size=self.batch_size)
                next_state_batch, loss = self.train_step(previous_state_batch)

            # todo: можно создать отдельный класс "experiments watcher"
            if step % 100 == 0:
                if self.jupyter:
                    clear_output()

                self.create_rule_imprint(train_step=step, pre_state=previous_state_batch, post_state=next_state_batch)
                tf.summary.scalar('loss_log', data=np.log10(loss), step=step)
                tf.summary.image("Example of CA figures", to_rgb(previous_state_batch[0])[None, ...], step=step)

            if step % 10 == 0:
                pool_states = self.petri_dish.set_of_petri_dishes
                pool_figures = generate_pool_figures(pool_states=pool_states,
                                                     train_step=step,
                                                     save_path=str(self.last_pictures_folder),
                                                     return_pool=True)

                tf.summary.image("Pool CA figures", pool_figures[None, ...], max_outputs=25, step=step)

            print(f"\r step: {step}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')
