from typing import Optional

import numpy as np
import tensorflow as tf  # TensorFlow version >= 2.0
from tensorflow.keras.layers import Conv2D


def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


class CAModel(tf.keras.Model):
    def __init__(self, channel_n: int, fire_rate: float):
        """
        Последовательность операции следующая:
            На вход: [bs, rows, cols, channels] (channels last)
            Первая свертка: (128 фильтров, 1 - поле восприимчивости (не совсем ясно как тогда они смотрят на соседей))
            Выход первой свертки: [bs, rows, cols, 128]
            Вторая свертка: (channels - фильтров, 1 - поле восприятия)
            Выход второй свертки: [bs, rows, cols, 16] - он сравнивается с обучающимися примерами

        Я так понимаю что масштаб изображения сохраняется из-за того что поле восприятия (kernel_size) равен 1.

        Args:
            channel_n: длинна состояния отдельной ячейки
            fire_rate: ***; todo: дописать по статье
        """
        super(CAModel, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        # todo: почему инициализация нулями ?
        self.strange_model = tf.keras.Sequential([Conv2D(filters=128, kernel_size=1, activation=tf.nn.relu),
                                                  Conv2D(filters=self.channel_n, kernel_size=1, activation=None,
                                                         kernel_initializer=tf.zeros_initializer)])

    @tf.function
    def perceive(self, x, angle=0.0):
        """ Похоже что эта вся херня необучаемая """
        # get identify mask for single cell
        identify = np.float32([0, 1, 0])
        identify = np.outer(identify, identify)  # identify: [[000], [010], [000]];

        # calculate Sobel filter as kernel for single cell
        # TODO: почему делим на 8 ?
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # dx: Sobel filter 'X' value [[-1, -2, -1], [000], [1, 2, 1]]/8;
        dy = dx.T  # dx: Sobel filter 'X' value [[1, 2, 1], [000], [-1, -2, -1]]/8;
        c, s = tf.cos(angle), tf.sin(angle)
        kernel = tf.stack([identify, c * dx - s * dy, c * dy + s * dx], -1)  # kernel shape: [3, 3, 3]
        # А это ВИДИМО новый способ управлять осями тензоров в tf 2.*
        kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, None, 3]
        kernel = tf.repeat(input=kernel, repeats=self.channel_n, axis=2)  # kernel shape: [3, 3, self.channel_n, 3]

        # x shape: [Batch, Height, Width, Channels]
        y = tf.nn.depthwise_conv2d(input=x, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        # y shape: [Batch, Height, Width, self.channel_n * 3]

        return y

    @tf.function
    def call(self, ca_state, fire_rate: Optional[float] = None, angle: float = 0.0, step_size: float = 1.0):
        """
        Здесь имплементирован один шаг обновления клеточного автомата.
        Args:
            ca_state: tf.tensor with shape = (batch, height, width, channels); it is our batch of cellar automata
                      states;
            fire_rate: probability of ***; todo: дописать по статье
            angle: angle of transformation on our target image; todo: надо проверить по статье так ли это
            step_size: ***; todo: дописать по статье

        Returns:
            Новое состояние клеточного автомата.
        """
        pre_life_mask = self.get_living_mask(ca_state)  # pre_life_mask shape: [Batch, Height, Width, 1];

        y = self.perceive(ca_state, angle)  # y shape: [Batch, Height, Width, self.channel_n * 3]
        ca_delta = self.strange_model(y) * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate

        # TODO: почему мы используем рандом ?
        update_mask = tf.random.uniform(tf.shape(ca_state[:, :, :, :1])) <= fire_rate
        # TODO: check that update_mask shape: [Batch, Height, Width, 1]
        ca_state += ca_delta * tf.cast(update_mask, tf.float32)

        post_life_mask = self.get_living_mask(ca_state)
        life_mask = pre_life_mask & post_life_mask

        return ca_state * tf.cast(life_mask, tf.float32)

    @staticmethod
    def get_living_mask(x):
        """
        Берет из общего тензора срез по оси каналов. А именно матрицу стоящую под индексом 3. Эта матрица, или этот
        канал отвечает за маску отобрадающую состояния "живых клеток" в клеточном автомате.

        Args:
            x: tf.tensor with shape = (batch, height, width, channels); it is our batch of cellar automata states;

        Returns:
            living_mask with shape: [Batch, Height, Width, 1]; заполнена нулями и единицами;
        """
        alpha = x[:, :, :, 3:4]  # alpha shape: [Batch, Height, Width, 1]
        pool_result = tf.nn.max_pool2d(input=alpha, ksize=3, strides=[1, 1, 1, 1], padding='SAME')
        living_mask = pool_result > 0.1  # living_mask shape: [Batch, Height, Width, 1]; заполнена нулями и единицами;

        return living_mask
