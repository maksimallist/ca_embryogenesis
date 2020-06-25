import io
from pathlib import Path
from typing import Optional
from typing import Union

import PIL.Image
import PIL.ImageDraw
import numpy as np
import requests
import tensorflow as tf
from IPython.display import Image, display

from embryogenesis.core.video_writer import tile2d


# todo: эта функция может быть и обычной, удалить tf отсюда
@tf.function
def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


# ------------------------------------------------- Image functions ----------------------------------------------------
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


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def image_save(path: Union[str, io.BytesIO], image_array: np.array, image_format: Optional[str] = None):
    image_array = np.asarray(image_array)

    if not image_format:
        image_format = path.rsplit('.', 1)[-1].lower()
        if image_format == 'jpg':
            image_format = 'jpeg'

    path = open(path, 'wb')
    np2pil(image_array).save(path, image_format, quality=95)


# ------------------------------------------- Jupyter Notebook Image functions -----------------------------------------
def image_encode(image_array: np.array, image_format: str = 'jpeg'):
    image_array = np.asarray(image_array)
    if len(image_array.shape) == 3 and image_array.shape[-1] == 4:
        image_format = 'png'
    f = io.BytesIO()
    image_save(f, image_array, image_format)
    return f.getvalue()


def image_show(image_array: np.array, image_format='jpeg'):
    # display(Image(data=image_encode(image_array, image_format)))
    display_tuple = (Image(data=image_encode(image_array, image_format)),)
    display(display_tuple)


# ------------------------------------------------- Visualise CA Results -----------------------------------------------
def generate_pool_figures(pool_states: np.array,
                          train_step: int,
                          save_path: str,
                          return_pool: bool = False) -> Union[np.array, None]:
    tiled_pool = tile2d(to_rgb(pool_states[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]

    image_save(save_path + '/%04d_pool.jpg' % train_step, tiled_pool)

    if return_pool:
        return np.asarray(tiled_pool)


def visualize_batch(pre_state: np.array,
                    post_state: np.array,
                    train_step: int,
                    save_path: str,
                    jupyter: bool = False) -> None:
    vis0 = np.hstack(to_rgb(pre_state).numpy())
    vis1 = np.hstack(to_rgb(post_state).numpy())
    vis = np.vstack([vis0, vis1])
    # save pictures
    image_save(save_path + '/batches_%04d.jpg' % train_step, vis)
    # visualize pictures in notebook or operation system
    if jupyter:
        print('batch (before/after):')
        image_show(vis)
