from typing import Tuple, Optional, Union

import numpy as np
import tensorflow as tf  # TensorFlow version >= 2.0
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer

from embryogenesis.model.utils import load_emoji, to_rgba


class StateObservation(Layer):
    def __init__(self,
                 channel_n: int,
                 norm_value: int = 8,
                 name='perception kernel',
                 **kwargs):
        super(StateObservation, self).__init__(name=name, **kwargs)
        self.channel_n = channel_n
        self.norm_value = norm_value

    def call(self, inputs, **kwargs):
        state_tensor, angle = inputs

        # get identify mask for single cell
        identify_mask = tf.constant([0., 1.0, 0.], dtype=tf.float32)
        identify_mask = tf.tensordot(identify_mask, identify_mask, axes=0)  # identify: [[000], [010], [000]];

        # calculate Sobel filter as kernel for single cell
        # Уточнение dx: Sobel filter 'X' value [[-1, -2, -1], [000], [1, 2, 1]]/8;
        x_1 = tf.constant([1.0, 2.0, 1.0], dtype=tf.float32)
        x_2 = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
        dx = tf.tensordot(x_1, x_2, axes=0) / self.norm_value  # todo: почему делим на 8 ?
        dy = tf.transpose(dx)  # dx: Sobel filter 'X' value [[1, 2, 1], [000], [-1, -2, -1]]/8;

        c, s = tf.cos(angle), tf.sin(angle)

        kernel = tf.stack([identify_mask, c * dx - s * dy, c * dy + s * dx], -1)  # kernel shape: [3, 3, 3]
        # А это видимо новый способ управлять осями тензоров в tf 2.*; таким образом можно увеличить размерность тензора
        kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, None, 3]
        kernel = tf.repeat(input=kernel, repeats=self.channel_n, axis=2)  # kernel shape: [3, 3, self.channel_n, 3]

        # perceive neighbors cells states
        observation = tf.nn.depthwise_conv2d(input=state_tensor,
                                             filter=kernel,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME')  # shape: [Batch, Height, Width, self.channel_n * 3]

        return observation


class LivingMask(Layer):
    def __init__(self,
                 life_threshold: float,  # 0.1
                 left_border: int = 3,
                 right_border: int = 4,
                 kernel_size: int = 3,
                 name='get living mask',
                 **kwargs):
        super(LivingMask, self).__init__(name=name, **kwargs)
        self.life_threshold = life_threshold
        self.left_border = left_border
        self.right_border = right_border
        self.kernel_size = kernel_size

    def call(self, inputs, **kwargs):
        living_slice = inputs[:, :, :, self.left_border:self.right_border]  # alpha shape: [Batch, Height, Width, 1]
        pool_result = tf.nn.max_pool2d(input=living_slice, ksize=self.kernel_size, strides=[1, 1, 1, 1], padding='SAME')
        # living_mask shape: [Batch, Height, Width, 1]; заполнена нулями и единицами;
        living_mask = pool_result > self.life_threshold

        return living_mask


class UpdateRule(Model):
    def __init__(self,
                 name: str,
                 fire_rate: float,
                 life_threshold: float,  # 0.1
                 channel_n: int,
                 conv_1_filters: int = 128,
                 conv_kernel_size: int = 1,
                 step_size: int = 1,
                 **kwargs):
        super(UpdateRule, self).__init__(name=name, **kwargs)
        self.fire_rate = tf.cast(fire_rate, tf.float32)
        self.step_size = tf.cast(step_size, tf.float32)
        self.get_living_mask = LivingMask(life_threshold=life_threshold)
        self.observation = StateObservation(channel_n=channel_n)

        self.conv_1 = Conv2D(filters=conv_1_filters,
                             kernel_size=conv_kernel_size,
                             activation=tf.nn.relu)

        self.conv_2 = Conv2D(filters=channel_n,
                             kernel_size=conv_kernel_size,
                             activation=None,  # ??????????????
                             kernel_initializer=tf.zeros_initializer)  # ??????????????

    def call(self, inputs, **kwargs):
        petri_dish, angle = inputs

        pre_life_mask = self.get_living_mask(petri_dish)  # shape: [Batch, Height, Width, 1];
        state_observation = self.observation([petri_dish, angle])  # kernel shape: [3, 3, self.channel_n, 3]

        conv_out = self.conv_1(state_observation)
        ca_delta = self.conv_2(conv_out) * self.step_size

        # todo: почему мы используем рандом ?
        update_mask = tf.random.uniform(tf.shape(petri_dish[:, :, :, :1])) <= self.fire_rate
        petri_dish += ca_delta * tf.cast(update_mask, tf.float32)

        post_life_mask = self.get_living_mask(petri_dish)
        life_mask = pre_life_mask & post_life_mask

        new_state = petri_dish * tf.cast(life_mask, tf.float32)

        return new_state


class PetriDish:  # бывший SamplePool
    def __init__(self,
                 channel_n: int,
                 shape: Tuple[int, int],
                 pool_size: int = 1024,
                 live_state_axis: int = 3,
                 morph_axis: Tuple[int, int] = (0, 2)):
        # cells shape
        self.channel_n = channel_n
        self.height, self.width = shape

        # axis description
        self.live_state_axis = live_state_axis
        self.morph_axis = morph_axis
        self.additional_axis = (self.live_state_axis + 1, self.channel_n - 1)

        # main attributes
        self.pool_size = pool_size
        self.cells = None
        self.petri_dish = None

    def make_seed(self, return_seed: bool = False) -> Union[None, np.array]:
        self.cells = np.zeros([self.height, self.width, self.channel_n], np.float32)
        self.cells[self.height // 2, self.width // 2, self.morph_axis[1] + 1:] = 1.0

        if return_seed:
            return self.cells

    def create_petri_dish(self, return_dish: bool = False, pool_size: Optional[int] = None) -> Union[None, np.array]:
        self.make_seed()

        if pool_size and return_dish:
            return np.repeat(self.cells[None, ...], repeats=pool_size, axis=0)
        else:
            self.petri_dish = np.repeat(self.cells[None, ...], repeats=self.pool_size, axis=0)

    def sample(self, batch_size: int = 32) -> Tuple[np.array, np.array]:
        if not self.petri_dish:
            self.create_petri_dish()

        batch_idx = np.random.choice(self.pool_size, batch_size, False)
        batch = self.petri_dish[batch_idx]

        return batch, batch_idx

    def commit(self, batch_cells: np.array, cells_idx: np.array) -> None:
        self.petri_dish[cells_idx] = batch_cells


class UpdateRuleTrainer:
    def __init__(self,
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
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
        self.optimizer = tf.keras.optimizers.Adam(lr_sched)

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
        seed = self.petri_dish.make_seed(return_seed=True)
        loss_log = []

        for i in range(self.train_steps + 1):
            if self.use_pattern_pool:
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

            # step_i = len(loss_log)
            loss_log.append(loss.numpy())

            # if step_i % 10 == 0:
            #     generate_pool_figures(pool, step_i)
            # if step_i % 100 == 0:
            #     clear_output()
            #     visualize_batch(x0, x, step_i)
            #     plot_loss(loss_log)
            #     export_model(ca, 'train_log/%04d' % step_i)

            print(f"\r step: {len(loss_log)}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')
