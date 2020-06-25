import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer


class StateObservation(Layer):
    def __init__(self,
                 channel_n: int,
                 norm_value: int = 8,
                 name='perception_kernel',
                 **kwargs):
        super(StateObservation, self).__init__(name=name, **kwargs)
        self.channel_n = channel_n
        self.norm_value = norm_value

        # get identify mask for single cell
        identify_mask = tf.constant([0., 1.0, 0.], dtype=tf.float32)
        self.identify_mask = tf.tensordot(identify_mask, identify_mask, axes=0)  # identify: [[000], [010], [000]];
        # get Sobel filters for x and y axis
        self.sobel_filter_x, self.sobel_filter_y = self.create_sobel_filter()

    def create_sobel_filter(self):
        # calculate Sobel filter as kernel for single cell
        # Уточнение dx: Sobel filter 'X' value [[-1, -2, -1], [000], [1, 2, 1]]/8;
        x_1 = tf.constant([1.0, 2.0, 1.0], dtype=tf.float32)
        x_2 = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
        sobel_filter_x = tf.tensordot(x_1, x_2, axes=0) / self.norm_value  # todo: почему делим на 8 ?
        sobel_filter_y = tf.transpose(sobel_filter_x)  # dx: Sobel filter 'X' value [[1, 2, 1], [000], [-1, -2, -1]]/8;

        return sobel_filter_x, sobel_filter_y

    def call(self, inputs, **kwargs):
        c, s = tf.cos(0.0), tf.sin(0.0)  # angle

        kernel = tf.stack([self.identify_mask,
                           c * self.sobel_filter_x - s * self.sobel_filter_y,
                           c * self.sobel_filter_y + s * self.sobel_filter_x], -1)  # kernel shape: [3, 3, 3]
        kernel = kernel[:, :, None, :]  # kernel shape: [3, 3, 1, 3]
        kernel = tf.repeat(input=kernel, repeats=self.channel_n, axis=2)  # kernel shape: [3, 3, self.channel_n, 3]

        # perceive neighbors cells states
        observation = tf.nn.depthwise_conv2d(input=inputs,
                                             filter=kernel,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME')  # shape: [Batch, Height, Width, self.channel_n * 3]

        return observation


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