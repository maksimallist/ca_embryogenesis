from pathlib import Path
from typing import Optional, Union

import tqdm
from tensorflow.keras.models import load_model

from core.image_utils import to_rgb
from core.petri_dish import PetriDish
from core.video_writer import VideoWriter, tile2d, zoom


class MorphCA:
    """
    The class of reproducing functional of a cellular automaton. That is, the class contains both a set of cells and
    their states, and cell renewal rules. Also, for this class, methods are implemented that allow you to apply update
    rules to a set of cells to obtain a new set of their states, thereby simulating the process of developing
    the entire system for a specified number of steps. One step in the simulation is the one-time application of the
    update rule to the cell set, and the replacement of the old set of cell states with a new one.
    """
    def __init__(self,
                 rule_model_path: Union[str, Path],
                 write_video: bool = False,
                 video_name: Optional[str] = None,
                 save_video_path: Optional[Union[str, Path]] = None,
                 print_summary: bool = True):
        """
        Loading the tensorflow checkpoints with neural network that determine cellar automaton update rule. And create
        instance if class PetriDish that determine functional of cells set and their states.

        Args:
            rule_model_path: path to tensorflow checkpoint
            write_video: boolean trigger that determine write the video with cellar automaton growth or not
            video_name: name of the video file
            save_video_path: save path for video file, if value is None, file will be saved in local folder
            print_summary: determine print Keras network summary or not
        """
        self.rule = load_model(rule_model_path)
        if print_summary:
            self.rule.summary()
        # TODO: исправить недорозумения со слоями inputs
        # _, height, width, channel_n = self.rule.input.shape
        height, width, channel_n = 72, 72, 16

        self.petri_dish = PetriDish(height=height, width=width, channel_n=channel_n)
        self.seed = self.petri_dish.make_seed(return_seed=True)[None, ...]

        self.write_video = write_video
        if write_video:
            if isinstance(save_video_path, str):
                save_video_path = Path(save_video_path)

            if video_name:
                video_name = video_name.split('.')

                if len(video_name) == 1:
                    video_name = video_name[0] + '.mp4'
                else:
                    if video_name[-1] != 'mp4':
                        video_name = video_name[0] + '.mp4'

                save_video_path = save_video_path.joinpath(video_name)
            else:
                save_video_path = save_video_path.joinpath('mca_grow.mp4')

            self.video_writer = VideoWriter(str(save_video_path))

    def step(self, state):
        """
        Once applies the update rule to a set of cells states.

        Args:
            state: numpy tensor with shape [batch_size, height, width, channel_n] containing states of cells

        Returns:
            New sells states
        """
        return self.rule(state)

    def run_growth(self, steps: int, return_state: bool = False):
        """
        Run simulations of growth of cellular automata.

        Args:
            steps: number of simulation steps
            return_state: trigger that determine return final cell states or not

        Returns:
            None or set of cell states
        """
        if self.write_video:
            with self.video_writer as video:
                video.add(zoom(tile2d(to_rgb(self.seed), 5), 2))
                state = self.seed
                # grow
                for _ in tqdm.trange(steps):
                    state = self.rule(state)
                    video.add(zoom(tile2d(to_rgb(state), 5), 2))
        else:
            state = self.seed
            # grow
            for _ in tqdm.trange(steps):
                state = self.rule(state)

        if return_state:
            return state

    def save_state(self):
        # todo: добавить функцию сохранения состояния клеточного автомата как картинки, или как тензора
        pass
