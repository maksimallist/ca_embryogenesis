import numpy as np
import tensorflow as tf  # TensorFlow version >= 2.0
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer
from typing import Tuple

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
    def __init__(self, channel_n: int,
                 shape: Tuple[int, int],
                 live_state_axis: int = 3,
                 morph_axis: Tuple[int, int] = (0, 2),
                 *, _parent=None, _parent_idx=None, **slots):

        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None

        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))



        self.channel_n = channel_n
        self.height, self.width = shape

        self.live_state_axis = live_state_axis
        self.morph_axis = morph_axis
        self.additional_axis = (self.live_state_axis + 1, self.channel_n - 1)

        self.cells_seed = self.make_seed()

    def make_seed(self):
        cells = np.zeros([self.height, self.width, self.channel_n], np.float32)
        cells[self.height // 2, self.width // 2, self.morph_axis[1] + 1:] = 1.0
        return cells

    # todo: убрать отсюда tensorflow
    @tf.function
    def make_circle_masks(self, n: int):
        """ Для повреждений в демо """
        x = tf.linspace(-1.0, 1.0, self.width)[None, None, :]
        y = tf.linspace(-1.0, 1.0, self.height)[None, :, None]

        center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
        r = tf.random.uniform([n, 1, 1], 0.1, 0.4)

        x, y = (x - center[0]) / r, (y - center[1]) / r
        mask = tf.cast(x * x + y * y < 1.0, tf.float32)

        return mask

    def sample(self, batch_size: int):
        idx = np.random.choice(self._size, batch_size, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = PetriDish(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)
