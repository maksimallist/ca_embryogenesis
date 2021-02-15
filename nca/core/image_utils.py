import io
from pathlib import Path
from typing import Optional, Union

import PIL.Image
import PIL.ImageDraw
import numpy as np
import requests
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


def to_alpha(x: np.array, life_state_axis: int = 3) -> np.array:
    return np.clip(x[..., life_state_axis:life_state_axis + 1], a_min=0.0, a_max=1.0)


def to_rgb(x: np.array, life_state_axis: int = 3) -> np.array:
    rgb = x[..., :life_state_axis]
    life_mask = to_alpha(x, life_state_axis)
    return 1.0 - life_mask + rgb


def zoom(img: np.array, scale: int = 4) -> np.array:
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def tile2d(a: np.array, w: Optional[int] = None) -> np.array:
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


class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)

        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)

        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)

        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)

        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


def get_image(img, max_size: int):
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # premultiply RGB by Alpha (only for images with transparent layer)
    img[..., :3] *= img[..., 3:]

    return img


def open_image(path: Union[str, Path], max_size: int):
    img = PIL.Image.open(path)
    return get_image(img=img, max_size=max_size)


def load_emoji(emoji, max_size):
    code = hex(ord(emoji))[2:].lower()
    url = f"https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u{code}.png"
    req = requests.get(url)
    img = PIL.Image.open(io.BytesIO(req.content))

    return get_image(img=img, max_size=max_size)
