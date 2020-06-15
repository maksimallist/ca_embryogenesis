from pathlib import Path
from typing import Optional, Union

import numpy as np
import tqdm
from tensorflow.keras.models import load_model

from embryogenesis.model.petri_dish import PetriDish
from embryogenesis.model.video_writer import VideoWriter
from embryogenesis.model.visualize_functions import to_rgb


def tile2d(a, w=None):
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


class MorphCA:
    def __init__(self,
                 rule_model_path: Union[str, Path],
                 write_video: bool = False,
                 video_name: Optional[str] = None,
                 save_video_path: Optional[Union[str, Path]] = None):
        self.rule = load_model(rule_model_path)
        _, height, width, channel_n = self.rule.input_layer.shape
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
