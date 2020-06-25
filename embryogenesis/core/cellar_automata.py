from pathlib import Path
from typing import Optional, Union

import tqdm
from tensorflow.keras.models import load_model

from embryogenesis.core.petri_dish import PetriDish
from embryogenesis.core.video_writer import VideoWriter, tile2d, zoom
from embryogenesis.core.image_utils import to_rgb


class MorphCA:
    def __init__(self,
                 rule_model_path: Union[str, Path],
                 write_video: bool = False,
                 video_name: Optional[str] = None,
                 save_video_path: Optional[Union[str, Path]] = None,
                 print_summary: bool = True):
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
        return self.rule(state)

    def run_growth(self, steps: int, return_state: bool = False):
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
        pass