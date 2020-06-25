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
        # Petri dish shape
        self.height, self.width, self.channel_n = height, width, channel_n

        # axis description
        self.image_axis = image_axis
        self.live_state_axis = live_state_axis
        self.additional_axis = (self.live_state_axis + 1, self.channel_n - 1)

        # main attributes
        self.pool_size = pool_size
        self.petri_dish = None
        self.set_of_petri_dishes = None

    def make_seed(self, return_seed: bool = False) -> Union[None, np.array]:
        self.petri_dish = np.zeros([self.height, self.width, self.channel_n], np.float32)
        # put one life cell in center of dish
        self.petri_dish[self.height // 2, self.width // 2, self.image_axis[1] + 1:] = 1.0
        if return_seed:
            return self.petri_dish

    def create_petri_dishes(self, pool_size: Optional[int] = None, return_dish: bool = False) -> Union[None, np.array]:
        self.make_seed()
        if pool_size and return_dish:
            return np.repeat(self.petri_dish[None, ...], repeats=pool_size, axis=0)
        else:
            # shape of petri_dish [pool_size, height, width, channel_n]
            self.set_of_petri_dishes = np.repeat(self.petri_dish[None, ...], repeats=self.pool_size, axis=0)

    def sample(self, batch_size: int = 32) -> Tuple[np.array, np.array]:
        if self.set_of_petri_dishes is None:
            self.create_petri_dishes()

        batch_idx = np.random.choice(self.pool_size, size=batch_size, replace=False)
        batch = self.set_of_petri_dishes[batch_idx]

        return batch, batch_idx

    def commit(self, batch_cells: np.array, cells_idx: np.array) -> None:
        self.set_of_petri_dishes[cells_idx] = batch_cells