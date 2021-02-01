import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer


class StateObservation(Layer):
    def __init__(self,
                 channel_n: int,
                 norm_value: int = 8,
                 observation_angle: float = 0.0,
                 name='perception_kernel',
                 **kwargs):
        super(StateObservation, self).__init__(name=name, **kwargs)
        # get identify mask for single cell
        self.identify_mask = tf.constant([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0]], dtype=tf.float32)

        # create Sobel operators for 'x' and 'y' axis
        sobel_filter_x = tf.constant([[-1.0, 0.0, 1.0],
                                      [-2.0, 0.0, 2.0],
                                      [-1.0, 0.0, 1.0]], dtype=tf.float32) / norm_value

        sobel_filter_y = tf.constant([[-1.0, -2.0, -1.0],
                                      [0.0, 0.0, 0.0],
                                      [1.0, 2.0, 1.0]], dtype=tf.float32) / norm_value

        # create kernel for depthwise_conv2d layer
        observation_angle = tf.constant(observation_angle, dtype=tf.float32)
        # kernel shape: [3, 3, 3]
        kernel = tf.stack([self.identify_mask,
                           tf.cos(observation_angle) * sobel_filter_x - tf.sin(observation_angle) * sobel_filter_y,
                           tf.cos(observation_angle) * sobel_filter_y + tf.sin(observation_angle) * sobel_filter_x], -1)

        kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, 1, 3]
        kernel = tf.repeat(input=kernel, repeats=channel_n, axis=2)  # kernel shape: [3, 3, channel_n, 3]

        # create perception layer
        self.perception = tf.nn.depthwise_conv2d(filter=kernel,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')  # shape: [Batch, Height, Width, channel_n * 3]

    def call(self, inputs, **kwargs):
        # perceive neighbors cells states
        return self.perception(input=inputs)


class LivingMask(Layer):
    def __init__(self,
                 life_threshold: float,
                 left_border: int = 3,
                 right_border: int = 4,
                 kernel_size: int = 3,
                 name='get_living_mask',
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
                 life_threshold: float,
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
                             kernel_initializer=tf.zeros_initializer())  # ??????????????

    def call(self, inputs, **kwargs):
        pre_life_mask = self.get_living_mask(inputs)  # shape: [Batch, Height, Width, 1];

        state_observation = self.observation(inputs)  # kernel shape: [3, 3, self.channel_n, 3]
        conv_out = self.conv_1(state_observation)
        ca_delta = self.conv_2(conv_out) * self.step_size

        # за счет накладывание случайной маски на чашку петри, симулируется стохастичность обновления состояния клеток,
        # то есть клетки обновляются не одновременно, а как бы со случайным интервалом.
        time_stochastic_mask = tf.random.uniform(tf.shape(inputs[:, :, :, :1])) <= self.fire_rate
        inputs += ca_delta * tf.cast(time_stochastic_mask, tf.float32)

        post_life_mask = self.get_living_mask(inputs)
        life_mask = pre_life_mask & post_life_mask
        new_state = inputs * tf.cast(life_mask, tf.float32)

        return new_state
