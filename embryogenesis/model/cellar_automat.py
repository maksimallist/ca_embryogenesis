import tensorflow as tf  # TensorFlow version >= 2.0
from tensorflow.keras.layers import Conv2D
import numpy as np
from typing import Dict


class CAModel(tf.keras.Model):
    def __init__(self, ca_config: Dict) -> None:
        super(CAModel, self).__init__()
        self.ca_config = ca_config

        # cells of "organism"
        ca_shape = (ca_config['']['height'], ca_config['']['width'], ca_config['']['main_properties'])
        self.petri_dish = np.zeros(ca_shape)
        # additional cells properties
        add_ca_shape = (ca_config['']['height'], ca_config['']['width'], ca_config['']['additional_properties'])
        self.trainable_cellar_attributes = np.zeros(add_ca_shape)

        # update rule attributes
        # todo: возможно создание едра придется вынести в отдельный метод, чтобы мы могли вносить повороты
        self.perception_kernel = self._get_perception_kernel(angle=0.0)
        # trainable CNN model, that determine update common rule to all cells
        self.update_cnn = tf.keras.Sequential([Conv2D(filters=self.ca_config[''],
                                                      kernel_size=self.ca_config[''],
                                                      activation=tf.nn.relu),
                                               Conv2D(filters=self.ca_config[''],
                                                      kernel_size=self.ca_config[''],
                                                      activation=None,  # ??????????????
                                                      kernel_initializer=tf.zeros_initializer)])  # ??????????????

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
