from typing import Optional, Tuple

import numpy as np


class PetriDish:
    """
    Чашка петри представляет из себя сетку определенного масштаба, состоящую из отдельных "клеток". Масштаб/размер сетки
    задаётся аргументами "height" и "width", при объявлении объекта класса. Состояние одной клетки описывается вектором
    заданной длины. Три компоненты этого вектора отвечают за значения rgb каналов отдельного пикселя на изображении.
    Еще одна компонента отвечает за отслеживание того, какие клетки являются живыми, а какие нет. И соответственно
    принимает только значения 0/1. Оставшиеся компоненты не имеют явной интерпретации, их количество может быть
    произвольным.
    """

    def __init__(self,
                 height: int,
                 width: int,
                 cell_states: int,
                 rgb_axis: Tuple[int, int, int] = (0, 1, 2),
                 live_axis: int = 3,
                 print_summary: bool = True):
        self.height, self.width = height, width
        assert cell_states > 4
        self.channels = cell_states

        assert live_axis not in rgb_axis
        self.live_axis = live_axis
        self.rgb_axis = rgb_axis
        self.cells_tensor = np.zeros([self.height, self.width, self.channels], np.float32)
        if print_summary:
            self.summary()

    def summary(self):
        print(f"=================================== The cellar automata summary ===================================")
        print(f"The shape of cellar automata tensor: ({self.height}, {self.width}, {self.channels});")
        print(f"The indexes of rgb_axis: {self.rgb_axis}; The index of cells live status axis: {self.live_axis};")
        print(f"===================================================================================================")

    # todo: upgrade this method
    def cell_state_initialization(self, coordinates: Optional[Tuple[int, int]] = None) -> None:
        if coordinates is None:
            # put one life cell in center of dish
            self.cells_tensor[self.height // 2, self.width // 2, self.live_axis:] = 1.0
        else:
            x, y = coordinates
            assert x <= self.width
            assert y <= self.height
            self.cells_tensor[x, y, self.live_axis:] = 1.0

    def rebase(self):
        pass


class CADataGenerator:
    def __init__(self,
                 ca_tensor: np.array,
                 target: np.array,
                 set_size: int = 1024,
                 damage_n: Optional[int] = 3,
                 reseed_batch: bool = True):
        self.target = target
        self.seed = ca_tensor

        self.set_size = set_size
        self.damage_n = damage_n
        self.reseed_batch = reseed_batch
        self.target_shape = self.target.shape[:2]

        self.ca_set = np.repeat(ca_tensor[None, ...], repeats=set_size, axis=0)  # [pool_size, height, width, channel_n]

    def make_circle_damage_masks(self, n: int):
        x = np.linspace(-1.0, 1.0, self.target_shape[1])[None, None, :]
        y = np.linspace(-1.0, 1.0, self.target_shape[0])[None, :, None]

        center = np.random.uniform(-0.5, 0.5, (2, n, 1, 1))
        r = np.random.uniform(0.1, 0.4, (n, 1, 1))

        x, y = (x - center[0]) / r, (y - center[1]) / r
        circle = x * x + y * y
        mask = np.asarray(circle < 1.0, np.float32)

        return mask

    @staticmethod
    def metric(batch_x: np.array, batch_y: np.array):
        return np.mean(np.square(batch_x[..., :4] - batch_y), (-2, -3, -1))

    def sample(self, batch_size: int = 32) -> Tuple[np.array, np.array]:
        batch_idx = np.random.choice(self.set_size, size=batch_size, replace=False)
        batch = self.ca_set[batch_idx]

        # stabilize training process on start; prevent the equivalent of “catastrophic forgetting”;
        if self.reseed_batch:
            loss_rank = self.metric(batch, self.target).argsort()[::-1]
            batch, batch_idx = batch[loss_rank], batch_idx[loss_rank]
            batch[:1] = self.seed

        if self.damage_n:
            damage = 1.0 - self.make_circle_damage_masks(self.damage_n)[..., None]
            batch[-self.damage_n:] *= damage

        return (batch, batch_idx), self.target

    def commit(self, batch_cells: np.array, cells_idx: np.array) -> None:
        self.ca_set[cells_idx] = batch_cells
