import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Layer, Input


class StateObservation(Layer):
    def __init__(self, *args, **kwargs):
        super(StateObservation, self).__init__(name='perception_kernel', *args, **kwargs)

    def build(self, input_shape):
        state_tensor_shape, angle_shape = input_shape
        self.channel_n = state_tensor_shape[-1]  # tf.constant(state_tensor_shape[-1], dtype=tf.float32)

        # get identify mask for single cell
        identify_mask = tf.constant([0., 1.0, 0.], dtype=tf.float32)
        self.identify_mask = tf.tensordot(identify_mask, identify_mask, axes=0)  # identify: [[000], [010], [000]];

        # calculate Sobel filter as kernel for single cell
        # Уточнение dx: Sobel filter 'X' value [[-1, -2, -1], [000], [1, 2, 1]]/8;
        x_1 = tf.constant([1.0, 2.0, 1.0], dtype=tf.float32)
        x_2 = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
        # get normal value for filter
        norm_value = tf.constant(8.0, dtype=tf.float32)
        self.dx = tf.tensordot(x_1, x_2, axes=0) / norm_value  # todo: почему делим на 8 ?
        self.dy = tf.transpose(self.dx)  # dx: Sobel filter 'X' value [[1, 2, 1], [000], [-1, -2, -1]]/8;

    @tf.function
    def call(self, inputs, **kwargs):
        state_tensor, angle = inputs

        c, s = tf.cos(angle), tf.sin(angle)
        kernel = tf.stack([self.identify_mask,
                           c * self.dx - s * self.dy,
                           c * self.dy + s * self.dx], -1)  # kernel shape: [3, 3, 3]

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
                 life_threshold: float,
                 left_border: int = 3,
                 right_border: int = 4,
                 kernel_size: int = 3,
                 *args,
                 **kwargs):
        super(LivingMask, self).__init__(name='get_living_mask', *args, **kwargs)
        self.life_threshold = life_threshold
        self.left_border = left_border
        self.right_border = right_border
        self.kernel_size = kernel_size

    @tf.function
    def call(self, inputs, **kwargs):
        living_slice = inputs[:, :, :, self.left_border:self.right_border]  # alpha shape: [Batch, Height, Width, 1]
        pool_result = tf.nn.max_pool2d(input=living_slice, ksize=self.kernel_size, strides=[1, 1, 1, 1], padding='SAME')
        # living_mask shape: [Batch, Height, Width, 1]; заполнена нулями и единицами;
        living_mask = pool_result > self.life_threshold

        return living_mask

# inputs = keras.Input(shape=(784,), name='digits')
# x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
# x = layers.Dense(64, activation='relu', name='dense_2')(x)
# outputs = layers.Dense(10, name='predictions')(x)
#
# model = keras.Model(inputs=inputs, outputs=outputs)


def get_rule_model(height: int,
                   width: int,
                   channels: int,
                   fire_rate: float,
                   life_threshold: float,
                   channel_n: int,
                   conv_1_filters: int = 128,
                   conv_kernel_size: int = 1,
                   step_size: int = 1,
                   name: str = 'rule_model'):
    fire_rate = tf.cast(fire_rate, tf.float32)
    step_size = tf.cast(step_size, tf.float32)

    input_state = Input(shape=(height, width, channels), name='digits')
    input_angle = Input(shape=(1,), name='digits')

    get_living_mask = LivingMask(life_threshold=life_threshold)
    observation = StateObservation()
    conv_1 = Conv2D(filters=conv_1_filters,
                    kernel_size=conv_kernel_size,
                    activation=tf.nn.relu)
    conv_2 = Conv2D(filters=channel_n,
                    kernel_size=conv_kernel_size,
                    activation=None,  # ??????????????
                    kernel_initializer=tf.zeros_initializer())  # ??????????????
    # -------------------------------------------------------------------------------
    pre_life_mask = get_living_mask(input_state)  # shape: [Batch, Height, Width, 1];
    state_observation = observation([input_state, input_angle])  # kernel shape: [3, 3, self.channel_n, 3]

    conv_out = conv_1(state_observation)
    ca_delta = conv_2(conv_out) * step_size

    # todo: почему мы используем рандом ?
    update_mask = tf.random.uniform(tf.shape(input_state[:, :, :, :1])) <= fire_rate
    input_state += ca_delta * tf.cast(update_mask, tf.float32)

    post_life_mask = get_living_mask(input_state)
    life_mask = pre_life_mask & post_life_mask

    new_state = input_state * tf.cast(life_mask, tf.float32)
    # -------------------------------------------------------------------------------
    model = Model(inputs=[input_state, input_angle], outputs=new_state)

    return model
