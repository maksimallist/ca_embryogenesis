from typing import Tuple, Optional, Union

import numpy as np
import tensorflow as tf


class PetriDish:
    def __init__(self,
                 target_image: np.array,
                 target_padding: int,
                 channel_n: int,
                 pool_size: int = 1024,
                 live_state_axis: int = 3,
                 morph_axis: Tuple[int, int] = (0, 2)):
        # cells shape
        self.channel_n = channel_n
        self.target_image = tf.pad(target_image,
                                   [(target_padding, target_padding), (target_padding, target_padding), (0, 0)])
        self.height, self.width = self.target_image.shape[:2]

        # axis description
        self.live_state_axis = live_state_axis
        self.morph_axis = morph_axis
        self.additional_axis = (self.live_state_axis + 1, self.channel_n - 1)

        # main attributes
        self.pool_size = pool_size
        self.cells = None
        self.petri_dish = None

    def return_target(self):
        return self.target_image.numpy()

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
        if self.petri_dish is None:
            self.create_petri_dish()

        batch_idx = np.random.choice(self.pool_size, batch_size, False)
        batch = self.petri_dish[batch_idx]

        return batch, batch_idx

    def commit(self, batch_cells: np.array, cells_idx: np.array) -> None:
        self.petri_dish[cells_idx] = batch_cells
