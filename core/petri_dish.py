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
                 live_axis: int = 3):
        self.height, self.width = height, width
        assert cell_states > 4
        self.channels = cell_states

        assert live_axis not in rgb_axis
        self.live_axis = live_axis
        self.rgb_axis = rgb_axis
        self.cells_tensor = np.zeros([self.height, self.width, self.channels], np.float32)
        self.cell_state_initialization()

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


class SetOfCellularAutomata:
    def __init__(self, ca_tensor: np.array, set_size: int = 1024):
        self.set_size = set_size
        # shape of ca_set [pool_size, height, width, channel_n]
        self.ca_set = np.repeat(ca_tensor[None, ...], repeats=set_size, axis=0)

    def sample(self, batch_size: int = 32) -> Tuple[np.array, np.array]:
        batch_idx = np.random.choice(self.set_size, size=batch_size, replace=False)
        batch = self.ca_set[batch_idx]

        return batch, batch_idx

    def commit(self, batch_cells: np.array, cells_idx: np.array) -> None:
        self.ca_set[cells_idx] = batch_cells
