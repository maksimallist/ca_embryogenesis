from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from tensorflow.keras import Model
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from embryogenesis.model.petri_dish import PetriDish
from embryogenesis.model.utils import to_rgba
from embryogenesis.model.visualize_functions import visualize_batch, generate_pool_figures, to_rgb


class UpdateRuleTrainer:
    def __init__(self,
                 root: Path,
                 exp_name: str,
                 petri_dish: PetriDish,
                 rule_model: Model,
                 target_image: np.array,
                 batch_size: int,
                 train_steps: int,
                 use_pattern_pool: Union[int, List[int]],
                 damage_n: Optional[int] = None,
                 learning_rate: float = 2e-3,
                 boundaries: int = 2000,
                 lr_multiplier: float = 0.1,
                 left_end_of_range: int = 64,
                 right_end_of_range: int = 96,
                 grad_norm_value: float = 1e-8,
                 jupyter: bool = False) -> None:
        # experiments infrastructure attributes
        self.root = root
        self.exp_name = exp_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoints_folder = None
        self.pictures_folder = None
        self.last_pictures_folder = None
        self.tensorboard_logs = None
        self.jupyter = jupyter

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

        self.left_end_of_range = left_end_of_range
        self.right_end_of_range = right_end_of_range
        self.grad_norm_value = grad_norm_value

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
    def train_step(self, input_tensor, angle=0.0):
        # sample random int from train steps range
        iter_n = tf.random.uniform([], self.left_end_of_range, self.right_end_of_range, tf.int32)

        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                input_tensor = self.trainable_rule([input_tensor, angle])
            loss = tf.reduce_mean(self.loss_f(input_tensor))

        grads = g.gradient(loss, self.trainable_rule.weights)
        grads = [g / (tf.norm(g) + self.grad_norm_value) for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.trainable_rule.weights))

        return input_tensor, loss

    def create_rule_imprint(self, train_step: int, pre_state: np.array, post_state: np.array):
        model_path = self.checkpoints_folder.joinpath(str(train_step))
        model_path.mkdir()

        # TODO: fix it !!!
        # self.trainable_rule.save(str(model_path))
        # export_model(self.trainable_rule, '%04d' % train_step, channel_n=16)

        # todo: вариант решения
        # from keras.layers import Input
        # from keras.models import Model
        #
        # newInput = Input(batch_shape=(1, 128, 128, 3))
        # newOutputs = oldModel(newInput)
        # newModel = Model(newInput, newOutputs)

        self.last_pictures_folder = self.pictures_folder.joinpath(str(train_step))
        self.last_pictures_folder.mkdir()
        visualize_batch(pre_state=pre_state, post_state=post_state, train_step=train_step,
                        save_path=str(self.last_pictures_folder), jupyter=self.jupyter)

    def train(self):
        for step in range(self.train_steps + 1):
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

            if step % 100 == 0:
                if self.jupyter:
                    clear_output()

                self.create_rule_imprint(train_step=step, pre_state=batch, post_state=x)
                tf.summary.scalar('loss_log', data=np.log10(loss), step=step)
                tf.summary.image("Example of CA figures", to_rgb(x[0])[None, ...], step=step)

            if step % 10 == 0:
                pool_states = self.petri_dish.petri_dish
                pool_figures = generate_pool_figures(pool_states=pool_states,
                                                     train_step=step,
                                                     save_path=str(self.last_pictures_folder),
                                                     return_pool=True)

                tf.summary.image("Pool CA figures", pool_figures[None, ...], max_outputs=25, step=step)

            print(f"\r step: {step}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')
