from typing import Callable, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

from nca.core import CADataGenerator
from nca.core.watchers import ExpWatcher


class TFCATrainer:
    def __init__(self,
                 data_generator: CADataGenerator,
                 model: Model,
                 optimizer: Optimizer,
                 watcher: ExpWatcher,
                 loss_function: Optional[Callable]):
        self.model = model  # compiled keras model
        self.watcher = watcher
        self.data_generator = data_generator

        self.optimizer = optimizer
        self.loss_function = loss_function

    def train_step(self, batch_size: int, grad_norm_value: float, grow_steps: Tuple[int, int]):
        iter_n = tf.random.uniform([], grow_steps[0], grow_steps[1], tf.int32)
        (batch_x, batch_ids), targets = self.data_generator.sample(batch_size)
        # batch_x: np.array; batch_ids: list indexes;

        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                batch_x = self.model(batch_x)

            loss = tf.reduce_mean(self.loss_function(batch_x[..., :4], targets))  # batch_x to rgba

        grads = g.gradient(loss, self.model.weights)
        grads = [g / (tf.norm(g) + grad_norm_value) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.weights))

        # insert new ca tensors in pool
        self.data_generator.commit(batch_cells=batch_x, cells_idx=batch_ids)

        return loss, batch_x

    def train(self, train_steps: int, batch_size: int, grad_norm_value: float, grow_steps: Tuple[int, int]) -> None:
        print("[ Start training ... ]")
        self.watcher.save_config()
        for step in range(1, train_steps + 1, 1):
            loss, next_state_batch = self.train_step(batch_size, grad_norm_value, grow_steps)
            self.watcher.log_train(step, loss, self.model, next_state_batch)
