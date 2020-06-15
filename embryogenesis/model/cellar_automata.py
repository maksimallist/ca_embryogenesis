from pathlib import Path
from typing import Optional, Union

import numpy as np
from tensorflow.keras.models import load_model

from embryogenesis.model.petri_dish import PetriDish


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
                 save_video_path: Optional[Union[str, Path]] = None):
        self.write_video = write_video
        self.save_video_path = save_video_path
        assert (self.save_video_path and self.write_video) is True, ValueError("")

        self.rule = load_model(rule_model_path)
        self.petri_dish = PetriDish(target_image=, target_padding=, channel_n=)
