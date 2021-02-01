from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer


class StateObservation(Layer):
    def __init__(self,
                 channel_n: int,
                 kernel_type: str = 'sobel',
                 custom_kernel: Optional[np.array] = None,
                 kernel_norm_value: int = 8,
                 observation_angle: float = 0.0,
                 name='perception_kernel',
                 **kwargs):
        super(StateObservation, self).__init__(name=name, **kwargs)
        self.channel_n = channel_n
        self.norm_value = tf.constant(kernel_norm_value, dtype=tf.float32)

        # calculate angle attributes
        self.observation_angle = tf.constant(observation_angle, dtype=tf.float32)
        self.angle_cos, self.angle_sin = tf.cos(self.observation_angle), tf.sin(self.observation_angle)

        # get identify mask for single cell
        self.identify_mask = tf.constant([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0]], dtype=tf.float32)

        # create kernel for depthwise_conv2d layer
        if kernel_type == 'sobel':
            kernel = self.create_sobel_kernel()
        elif kernel_type == 'scharr':
            kernel = self.create_scharr_kernel()
        elif kernel_type == 'custom':
            kernel = self.create_custom_kernel(dx_kernel=custom_kernel)
        else:
            raise ValueError(f"The 'kernel_type' argument must be ['sobel', 'scharr', 'custom'] or None, "
                             f"but {kernel_type} was found.")

        # create perception layer
        self.perception = tf.nn.depthwise_conv2d(filter=kernel,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')  # shape: [Batch, Height, Width, channel_n * 3]

    def create_sobel_kernel(self):
        # create Sobel operators for 'x' and 'y' axis
        sobel_filter_x = tf.constant([[1.0, 0.0, -1.0],
                                      [2.0, 0.0, -2.0],
                                      [1.0, 0.0, -1.0]], dtype=tf.float32) / self.norm_value

        sobel_filter_y = tf.constant([[1.0, 2.0, 1.0],
                                      [0.0, 0.0, 0.0],
                                      [-1.0, -2.0, -1.0]], dtype=tf.float32) / self.norm_value

        kernel = tf.stack([self.identify_mask,  # kernel shape: [3, 3, 3]
                           self.angle_cos * sobel_filter_x - self.angle_sin * sobel_filter_y,
                           self.angle_cos * sobel_filter_y + self.angle_sin * sobel_filter_x], -1)
        kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, 1, 3]
        kernel = tf.repeat(input=kernel, repeats=self.channel_n, axis=2)  # kernel shape: [3, 3, channel_n, 3]

        return kernel

    def create_scharr_kernel(self):
        # create Scharr operators for 'x' and 'y' axis
        scharr_filter_x = tf.constant([[3.0, 0.0, -3.0],
                                       [10.0, 0.0, -10.0],
                                       [3.0, 0.0, -3.0]], dtype=tf.float32) / self.norm_value

        scharr_filter_y = tf.constant([[3.0, 10.0, 3.0],
                                       [0.0, 0.0, 0.0],
                                       [-3.0, -10.0, -3.0]], dtype=tf.float32) / self.norm_value

        kernel = tf.stack([self.identify_mask,  # kernel shape: [3, 3, 3]
                           self.angle_cos * scharr_filter_x - self.angle_sin * scharr_filter_y,
                           self.angle_cos * scharr_filter_y + self.angle_sin * scharr_filter_x], -1)
        kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, 1, 3]
        kernel = tf.repeat(input=kernel, repeats=self.channel_n, axis=2)  # kernel shape: [3, 3, channel_n, 3]

        return kernel

    def create_custom_kernel(self, dx_kernel: Optional[np.array]):
        if dx_kernel is None:
            raise ValueError("If you want create your custom kernel than the layer argument 'dx_kernel' "
                             "must be not None, but it is.")
        else:
            if dx_kernel.shape != (3, 3):
                raise ValueError(f"Shape of custom kernel must be (3, 3), but {dx_kernel.shape} was found.")
            else:
                # create custom operators for 'x' and 'y' axis
                custom_filter_x = tf.constant(dx_kernel, dtype=tf.float32) / self.norm_value
                custom_filter_y = tf.constant(tf.transpose(dx_kernel), dtype=tf.float32) / self.norm_value

                kernel = tf.stack([self.identify_mask,  # kernel shape: [3, 3, 3]
                                   self.angle_cos * custom_filter_x - self.angle_sin * custom_filter_y,
                                   self.angle_cos * custom_filter_y + self.angle_sin * custom_filter_x], -1)
                kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, 1, 3]
                kernel = tf.repeat(input=kernel, repeats=self.channel_n, axis=2)  # kernel shape: [3, 3, channel_n, 3]

        return kernel

    def call(self, inputs, **kwargs):
        # perceive neighbors cells states
        return self.perception(input=inputs)


class LivingMask(Layer):
    def __init__(self,
                 life_threshold: float = 0.1,
                 live_axis: int = 3,
                 kernel_size: int = 3,
                 name='get_living_mask',
                 **kwargs):
        super(LivingMask, self).__init__(name=name, **kwargs)
        self.life_threshold = life_threshold
        self.live_axis = live_axis
        self.live_mask = tf.nn.max_pool2d(ksize=kernel_size, strides=[1, 1, 1, 1], padding='SAME')

    def call(self, inputs, **kwargs):
        pool_result = self.live_mask(input=inputs[:, :, :, self.live_axis])  # alpha shape: [Batch, Height, Width, 1]
        return pool_result > self.life_threshold  # [Batch, Height, Width, 1]; заполнена нулями и единицами;


class UpdateRule(Model):
    def get_config(self):
        pass

    def __init__(self,
                 name: str,
                 channel_n: int,
                 life_threshold: float,
                 conv_1_filters: int = 128,
                 conv_kernel_size: int = 1,
                 stochastic_update: bool = True,
                 fire_rate: Optional[float] = 0.5,
                 **kwargs):
        super(UpdateRule, self).__init__(name=name, **kwargs)
        # define the mode of cellar automata grow
        self.stochastic_update = stochastic_update
        if stochastic_update and fire_rate:
            self.fire_rate = tf.cast(fire_rate, tf.float32)
        elif stochastic_update and fire_rate is None:
            raise ValueError("If you want to train the cellar automata in stochastic mode, "
                             "the 'fire_rate' argument must be not None, but it is.")

        # define the network layers
        self.get_living_mask = LivingMask(life_threshold=life_threshold)
        self.observation = StateObservation(channel_n=channel_n)
        self.conv_1 = Conv2D(filters=conv_1_filters,
                             kernel_size=conv_kernel_size,
                             activation=tf.nn.relu)
        self.conv_2 = Conv2D(filters=channel_n,
                             kernel_size=conv_kernel_size,
                             activation=None,
                             kernel_initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        # gets changing cell states
        pre_life_mask = self.get_living_mask(inputs)  # shape: [Batch, Height, Width, 1];
        state_observation = self.observation(inputs)  # kernel shape: [3, 3, self.channel_n, 3]
        conv_out = self.conv_1(state_observation)
        ca_delta = self.conv_2(conv_out)

        # update the cells states
        if self.stochastic_update:
            # за счет накладывания случайной маски, симулируется стохастичность обновления состояния клеток,
            # то есть клетки обновляются не одновременно, а как бы со случайным интервалом.
            time_stochastic_mask = tf.random.uniform(tf.shape(inputs[:, :, :, :1])) <= self.fire_rate
            new_states = inputs + ca_delta * tf.cast(time_stochastic_mask, tf.float32)
        else:
            new_states = inputs + ca_delta

        # determine which cells became alive and which did not
        post_life_mask = self.get_living_mask(new_states)
        life_mask = pre_life_mask & post_life_mask
        new_states = new_states * tf.cast(life_mask, tf.float32)

        return new_states
