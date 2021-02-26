from typing import Dict, List, Optional, Tuple, Union

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
    init_mode = None
    coordinates = None
    state_tensor = None
    initialized = False

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
        if self.initialized:
            print(f"The cellar automata state is initialized;")
            print(f"The initialization mode is '{self.init_mode}';")
        else:
            print(f"The cellar automata state is not initialized;")
        print(f"===================================================================================================")

    def cell_state_initialization(self,
                                  mode: str = 'center',
                                  coordinates: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
                                  state_tensor: Optional[np.array] = None) -> None:
        self.init_mode = mode
        self.coordinates = coordinates
        self.state_tensor = state_tensor

        if self.initialized is True:
            self.rebase()

        if mode == 'center':
            # put one life cell in center of dish
            self.cells_tensor[self.height // 2, self.width // 2, self.live_axis:] = 1.0
        elif mode == 'cell_position':
            if coordinates:
                x, y = coordinates
                assert x <= self.width
                assert y <= self.height
                self.cells_tensor[x, y, self.live_axis:] = 1.0
            else:
                raise ValueError(f"The 'cell_position' mode is selected, but the 'coordinates' "
                                 f"argument is not specified.")
        elif mode == 'few_seeds':
            if coordinates:
                if isinstance(coordinates, List):
                    for seed in coordinates:
                        x, y = seed
                        assert x <= self.width
                        assert y <= self.height
                        self.cells_tensor[x, y, self.live_axis:] = 1.0
                else:
                    raise ValueError(f"The 'few_seeds' mode is selected. The 'coordinates' argument must be "
                                     f"List[Tuple[int, int], but '{type(coordinates)}' was found.")
            else:
                raise ValueError(f"The 'few_seeds' mode is selected, but the 'coordinates' "
                                 f"argument is not specified.")
        elif mode == 'tensor':
            if state_tensor:
                assert (self.height, self.width, self.channels) == state_tensor.shape
                self.cells_tensor = state_tensor
        else:
            raise ValueError(f"The mode of initialization must be in "
                             f"['center', 'cell_position', 'few_seeds', 'tensor'], but {mode} was found.")

        self.initialized = True

    def rebase(self):
        self.cells_tensor = np.zeros([self.height, self.width, self.channels], np.float32)

    def reseed(self):
        if self.initialized:
            self.cell_state_initialization(self.init_mode, self.coordinates, self.state_tensor)
        else:
            raise ValueError(f"The method 'rebase' cannot be called because the state of the cellular automaton "
                             f"is not initialized")


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


class TextPD:
    initialized: bool = False
    init_mode: Optional[str] = None
    coordinates: Optional[Union[Tuple[int, str], List[Tuple[int, str]]]] = None
    state_text: Optional[str] = None

    def __init__(self,
                 length: int,
                 cell_state_shape: int,
                 vocab: Union[str, List[str], Dict[str, int]],
                 padding: int = 0,
                 print_summary: bool = True):
        self.length = length
        self.padding = padding
        assert cell_state_shape >= 2, ValueError("")  # TODO: write comment text
        self.shape = (length + padding * 2, cell_state_shape)
        self.vocab = vocab
        self.ind2char = {val: key for key, val in vocab.items()}
        self.cells_tensor = np.zeros([length + padding * 2, cell_state_shape], np.float32)
        if print_summary:
            self.summary()

    def summary(self):
        print(f"================================== The text cellar automata summary ==================================")
        print(f"The shape of cellar text automata tensor: {self.shape};")
        if self.initialized:
            print(f"The text cellar automata state is initialized;")
            print(f"The initialization mode is '{self.init_mode}';")
        else:
            print(f"The text cellar automata state is not initialized;")
        print(f"===================================================================================================")

    def cell_state_initialization(self,
                                  mode: str = 'center',
                                  coordinates: Optional[Union[Tuple[int, str], List[Tuple[int, str]]]] = None,
                                  text: Optional[str] = None) -> None:
        self.init_mode = mode
        self.coordinates = coordinates
        self.state_text = text

        if self.initialized is True:
            self.rebase()

        if mode == 'center':
            # put one life cell in center of dish
            # TODO add choose random char
            self.cells_tensor[self.length // 2, 0] = self.vocab.get('x') / float(len(self.vocab))
            self.cells_tensor[self.length // 2, 1] = 1.0
        elif mode == 'cell_position':
            if coordinates:
                x, y = coordinates
                assert x <= self.length
                self.cells_tensor[x, 0] = self.vocab[y] / float(len(self.vocab))
                self.cells_tensor[x, 1] = 1.0
            else:
                raise ValueError(f"The 'cell_position' mode is selected, but the 'coordinates' "
                                 f"argument is not specified.")
        elif mode == 'few_seeds':
            if coordinates:
                if isinstance(coordinates, List):
                    for seed in coordinates:
                        x, y = seed
                        assert x <= self.length
                        self.cells_tensor[x, 0] = self.vocab[y] / float(len(self.vocab))
                        self.cells_tensor[x, 1] = 1.0
                else:
                    raise ValueError(f"The 'few_seeds' mode is selected. The 'coordinates' argument must be "
                                     f"List[Tuple[int, int], but '{type(coordinates)}' was found.")
            else:
                raise ValueError(f"The 'few_seeds' mode is selected, but the 'coordinates' "
                                 f"argument is not specified.")
        elif mode == 'text':
            # TODO добавить проверку наличия всех символов из текста в словаре
            if text:
                assert self.length >= len(text)
                for ind, char in enumerate(text):
                    self.cells_tensor[ind + self.padding, 0] = self.vocab[char] / float(len(self.vocab))
                    self.cells_tensor[ind + self.padding, 1] = 1.0
        else:
            raise ValueError(f"The mode of initialization must be in "
                             f"['center', 'cell_position', 'few_seeds', 'tensor'], but {mode} was found.")

        self.initialized = True

    def rebase(self):
        self.cells_tensor = np.zeros(self.shape, np.float32)

    def reseed(self):
        if self.initialized:
            self.cell_state_initialization(self.init_mode, self.coordinates, self.state_text)
        else:
            raise ValueError(f"The method 'rebase' cannot be called because the state of the cellular automaton "
                             f"is not initialized")

    def np2text(self):
        text_tensor = np.uint8(self.cells_tensor[..., :1].clip(0, 1) * len(self.vocab))
        text_tensor = np.reshape(text_tensor, (text_tensor.shape[0],))

        string = ''
        for x in list(text_tensor):
            string += self.ind2char[x]

        return string


class TextCAGenerator:
    def __init__(self, target: str, ca_seed: np.array, set_size: int = 1024, reseed_batch: bool = True):
        self.target = target
        self.seed = ca_seed
        self.set_size = set_size
        self.ca_set = np.repeat(ca_seed[None, ...], repeats=set_size, axis=0)  # [pool_size, length, channel_n]
        self.reseed_batch = reseed_batch

    @staticmethod
    def metric(batch_x: np.array, batch_y: np.array):
        return np.mean(np.square(batch_x[..., :2] - batch_y), (-2, -1))

    def sample(self, batch_size: int = 32) -> Tuple[np.array, np.array]:
        batch_idx = np.random.choice(self.set_size, size=batch_size, replace=False)
        batch = self.ca_set[batch_idx]

        # stabilize training process on start; prevent the equivalent of “catastrophic forgetting”;
        if self.reseed_batch:
            loss_rank = self.metric(batch, self.target).argsort()[::-1]
            batch, batch_idx = batch[loss_rank], batch_idx[loss_rank]
            batch[:1] = self.seed

        return (batch, batch_idx), self.target

    def commit(self, batch_cells: np.array, cells_idx: np.array) -> None:
        self.ca_set[cells_idx] = batch_cells
