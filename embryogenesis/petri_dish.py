from typing import Optional, Tuple, Union

import numpy as np


class PetriDish:
    def __init__(self,
                 height: int,
                 width: int,
                 channel_n: int,
                 pool_size: int = 1024,
                 live_state_axis: int = 3,
                 image_axis: Tuple[int, int] = (0, 2)):
        # cells shape
        self.height, self.width, self.channel_n = height, width, channel_n

        # axis description
        self.live_state_axis = live_state_axis
        self.image_axis = image_axis
        self.additional_axis = (self.live_state_axis + 1, self.channel_n - 1)

        # main attributes
        self.pool_size = pool_size
        self.cells = None
        self.petri_dish = None

    def make_seed(self, return_seed: bool = False) -> Union[None, np.array]:
        self.cells = np.zeros([self.height, self.width, self.channel_n], np.float32)
        # put one life cell in center of dish
        self.cells[self.height // 2, self.width // 2, self.image_axis[1] + 1:] = 1.0
        if return_seed:
            return self.cells

    def create_petri_dish(self, pool_size: Optional[int] = None, return_dish: bool = False) -> Union[None, np.array]:
        self.make_seed()
        if pool_size and return_dish:
            return np.repeat(self.cells[None, ...], repeats=pool_size, axis=0)
        else:
            # shape of petri_dish [pool_size, height, width, channel_n]
            self.petri_dish = np.repeat(self.cells[None, ...], repeats=self.pool_size, axis=0)

    def sample(self, batch_size: int = 32) -> Tuple[np.array, np.array]:
        if self.petri_dish is None:
            self.create_petri_dish()

        batch_idx = np.random.choice(self.pool_size, size=batch_size, replace=False)
        batch = self.petri_dish[batch_idx]

        return batch, batch_idx

    def commit(self, batch_cells: np.array, cells_idx: np.array) -> None:
        self.petri_dish[cells_idx] = batch_cells
