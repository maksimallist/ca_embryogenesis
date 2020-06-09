from typing import Dict

import numpy as np
import tensorflow as tf  # TensorFlow version >= 2.0
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer


class PerceptionKernel(Layer):
    def __init__(self,
                 channel_n: int,
                 norm_value: int = 8,
                 name='perception kernel',
                 **kwargs):
        super(PerceptionKernel, self).__init__(name=name, **kwargs)
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

        return kernel


class CAModel(Model):
    def __init__(self, ca_config: Dict) -> None:
        super(CAModel, self).__init__()
        self.ca_config = ca_config
        self.channel_n = self.ca_config['channel_n']
        self.fire_rate = self.ca_config['fire_rate']

        # cells of "organism"
        main_ca_shape = (ca_config['height'], ca_config['width'], ca_config['main_properties'])
        self.phenotype = np.zeros(main_ca_shape)
        # additional trainable cells properties
        additional_ca_shape = (ca_config['height'], ca_config['width'], ca_config['additional_properties'])
        self.cellar_properties = np.zeros(additional_ca_shape)
        # full tensor with phenotype and additional cells properties
        self.petri_dish = np.stack([self.phenotype, self.cellar_properties], axis=2)

        # trainable CNN model, that determine update common rule to all cells
        self.update_cnn = tf.keras.Sequential([Conv2D(filters=self.ca_config[''],
                                                      kernel_size=self.ca_config[''],
                                                      activation=tf.nn.relu),
                                               Conv2D(filters=self.ca_config[''],
                                                      kernel_size=self.ca_config[''],
                                                      activation=None,  # ??????????????
                                                      kernel_initializer=tf.zeros_initializer)])  # ??????????????

    @tf.function
    def get_living_mask(self):
        """
        Берет из общего тензора срез по оси каналов. А именно матрицу стоящую под индексом 3. Эта матрица, или этот
        канал отвечает за маску отобрадающую состояния "живых клеток" в клеточном автомате.

        Returns:
            living_mask with shape: [Batch, Height, Width, 1]; заполнена нулями и единицами;
        """
        # todo: убрать хардкодинг
        living_slice = self.petri_dish[:, :, :, 3:4]  # alpha shape: [Batch, Height, Width, 1]
        pool_result = tf.nn.max_pool2d(input=living_slice, ksize=3, strides=[1, 1, 1, 1], padding='SAME')
        living_mask = pool_result > 0.1  # living_mask shape: [Batch, Height, Width, 1]; заполнена нулями и единицами;

        return living_mask

    @tf.function
    def _get_perception_kernel(self, angle=0.0):
        # get identify mask for single cell
        identify_mask = np.float32([0, 1, 0])
        identify_mask = np.outer(identify_mask, identify_mask)  # identify: [[000], [010], [000]];

        # calculate Sobel filter as kernel for single cell
        # Уточнение dx: Sobel filter 'X' value [[-1, -2, -1], [000], [1, 2, 1]]/8;
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # todo: почему делим на 8 ?
        dy = dx.T  # dx: Sobel filter 'X' value [[1, 2, 1], [000], [-1, -2, -1]]/8;

        c, s = tf.cos(angle), tf.sin(angle)

        kernel = tf.stack([identify_mask, c * dx - s * dy, c * dy + s * dx], -1)  # kernel shape: [3, 3, 3]
        # А это видимо новый способ управлять осями тензоров в tf 2.*; таким образом можно увеличить размерность тензора
        kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, None, 3]
        kernel = tf.repeat(input=kernel, repeats=self.channel_n, axis=2)  # kernel shape: [3, 3, self.channel_n, 3]

        return kernel

    @tf.function
    def growth_step(self, angle: float = 0.0, step_size: float = 1.0):
        pre_life_mask = self.get_living_mask()  # shape: [Batch, Height, Width, 1];
        perception_kernel = self._get_perception_kernel(angle=angle)  # kernel shape: [3, 3, self.channel_n, 3]

        # perceive neighbors cells states
        observation = tf.nn.depthwise_conv2d(input=self.petri_dish,
                                             filter=perception_kernel,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME')  # shape: [Batch, Height, Width, self.channel_n * 3]
        # apply update rule
        ca_delta = self.update_cnn(observation) * step_size

        # todo: почему мы используем рандом ?
        update_mask = tf.random.uniform(tf.shape(self.petri_dish[:, :, :, :1])) <= self.fire_rate
        self.petri_dish += ca_delta * tf.cast(update_mask, tf.float32)

        post_life_mask = self.get_living_mask()
        life_mask = pre_life_mask & post_life_mask

        return self.petri_dish * tf.cast(life_mask, tf.float32)
