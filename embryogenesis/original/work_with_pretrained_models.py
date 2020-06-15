import PIL.ImageFont
import moviepy.editor as mvp
import numpy as np
import tensorflow as tf
import tqdm
from matplotlib import font_manager as fm


def get_model(emoji='🦋', fire_rate=0.5, use_pool=1, damage_n=3, run=0, prefix='models/', output='model'):
    path = prefix

    assert fire_rate in [0.5, 1.0]

    if fire_rate == 0.5:
        path += 'use_sample_pool_%d damage_n_%d ' % (use_pool, damage_n)
    elif fire_rate == 1.0:
        path += 'fire_rate_1.0 '

    code = hex(ord(emoji))[2:].upper()
    path += 'target_emoji_%s run_index_%d/08000' % (code, run)

    assert output in ['model', 'json']

    if output == 'model':
        ca = CAModel(channel_n=16, fire_rate=fire_rate)
        ca.load_weights(path)
        return ca
    elif output == 'json':
        return open(path + '.json', 'r').read()


# ----------------------------------------------- Teaser --------------------------------------------------------
# !wget -O models.zip 'https://github.com/google-research/self-organising-systems/blob/master/assets/growing_ca/models.zip?raw=true'
# !unzip -oq models.zip

EMOJI = '🦎😀💥👁🐠🦋🐞🕸🥨🎄'
atlas = np.hstack([load_emoji(e) for e in EMOJI])
# imshow(atlas)

models = [get_model(emoji, run=1) for emoji in EMOJI]

with VideoWriter('teaser.mp4') as vid:
    x = np.zeros([len(EMOJI), 64, 64, CHANNEL_N], np.float32)
    # grow
    for i in tqdm.trange(200):
        k = i // 20
        if i % 20 == 0 and k < len(EMOJI):
            x[k, 32, 32, 3:] = 1.0
        vid.add(zoom(tile2d(to_rgb(x), 5), 2))
        for ca, xk in zip(models, x):
            xk[:] = ca(xk[None, ...])[0]
    # damage
    mask = PIL.Image.new('L', (64 * 5, 64 * 2))
    draw = PIL.ImageDraw.Draw(mask)
    for i in tqdm.trange(400):
        cx, r = i * 3 - 20, 6
        y1, y2 = 32 + np.sin(i / 5 + np.pi) * 8, 32 + 64 + np.sin(i / 5) * 8
        draw.rectangle((0, 0, 64 * 5, 64 * 2), fill=0)
        draw.ellipse((cx - r, y1 - r, cx + r, y1 + r), fill=255)
        draw.ellipse((cx - r, y2 - r, cx + r, y2 + r), fill=255)
        x *= 1.0 - (np.float32(mask).reshape(2, 64, 5, 64)
                    .transpose([0, 2, 1, 3]).reshape(10, 64, 64, 1)) / 255.0
        if i < 200 or i % 2 == 0:
            vid.add(zoom(tile2d(to_rgb(x), 5), 2))
        for ca, xk in zip(models, x):
            xk[:] = ca(xk[None, ...])[0]
    # fade out
    last = zoom(tile2d(to_rgb(x), 5), 2)
    for t in np.linspace(0, 1, 30):
        vid.add(last * (1.0 - t) + t)

mvp.ipython_display('teaser.mp4', loop=True)

# ------------------------------------------------- Unstable Patterns --------------------------------------------
# !wget -O slider.png 'https://github.com/google-research/self-organising-systems/raw/master/assets/growing_ca/slider.png?raw=true'


font_fn = fm.findfont(fm.FontProperties())
font = PIL.ImageFont.truetype(font_fn, 20)

models = [get_model(ch, use_pool=0, damage_n=0) for ch in EMOJI]
fn = 'unstable.mp4'

with VideoWriter(fn) as vid:
    x = np.zeros([len(EMOJI), 64, 64, CHANNEL_N], np.float32)
    x[:, 32, 32, 3:] = 1.0
    # grow
    slider = PIL.Image.open("slider.png")
    for i in tqdm.trange(1000):
        if i < 200 or i % 5 == 0:
            vis = zoom(tile2d(to_rgb(x), 5), 4).clip(0, 1)
            vis_extended = np.concatenate((vis, np.ones((164, vis.shape[1], 3))), axis=0)
            im = np.uint8(vis_extended * 255)
            im = PIL.Image.fromarray(im)
            im.paste(slider, box=(20, vis.shape[0] + 20))
            draw = PIL.ImageDraw.Draw(im)
            p_x = (14 + (610 / 1000) * i) * 2.0
            draw.rectangle([p_x, vis.shape[0] + 20 + 55, p_x + 10, vis.shape[0] + 20 + 82], fill="#434343bd")
            vid.add(np.uint8(im))
        for ca, xk in zip(models, x):
            xk[:] = ca(xk[None, ...])[0]
    # fade out
    for t in np.linspace(0, 1, 30):
        vid.add(vis_extended * (1.0 - t) + t)

mvp.ipython_display(fn, loop=True)

# ---------------------------------------------------- Rotation --------------------------------------------------------
row_size = 4
models_of_interest = ["🦋", "🦎", "🐠", "😀"]
num_images = 16
imgs = []
start_angle = np.random.randint(13, 76)

for i in np.arange(num_images):
    ang = start_angle + i * np.random.randint(36, 111)
    ang = ang / 360.0 * 2 * np.pi
    if i % row_size == 0:
        ca = get_model(models_of_interest[i // row_size])
    x = np.zeros([1, 56, 56, CHANNEL_N], np.float32)
    x[:, 28, 28, 3:] = 1.0
    for i in range(500):
        ang = tf.constant(ang, tf.float32)
        x = ca(x, angle=ang)
    imgs.append(to_rgb(x)[0])

# Assumes the result is a multiple of row_size
assert len(imgs) % row_size == 0
imgs = zip(*(iter(imgs),) * row_size)

imgs_arr = np.concatenate([np.hstack(im_row) for im_row in imgs])
vis = zoom(imgs_arr, 4)

imshow(vis, fmt='png')

# ------------------------------------------- Regeneration (trained without damage) ------------------------------------
models = [get_model(ch, damage_n=0) for ch in '😀🦋🦎']

with VideoWriter('regen1.mp4') as vid:
    x = np.zeros([len(models), 5, 56, 56, CHANNEL_N], np.float32)
    cx, cy = 28, 28
    x[:, :, cy, cx, 3:] = 1.0
    for i in tqdm.trange(2000):
        if i == 200:
            x[:, 0, cy:] = x[:, 1, :cy] = 0
            x[:, 2, :, cx:] = x[:, 3, :, :cx] = 0
            x[:, 4, cy - 8:cy + 8, cx - 8:cx + 8] = 0
        vis = to_rgb(x)
        vis = np.vstack([np.hstack(row) for row in vis])
        vis = zoom(vis, 2)
        if (i < 400 and i % 2 == 0) or i % 8 == 0:
            vid.add(vis)
        if i == 200:
            for _ in range(29):
                vid.add(vis)
        for ca, row in zip(models, x):
            row[:] = ca(row)

mvp.ipython_display('regen1.mp4')

# ------------------------------------------- Regeneration (trained with damage) ---------------------------------------
models = [get_model(ch, damage_n=3) for ch in '😀🦋🦎']

with VideoWriter('regen2.mp4') as vid:
    x = np.zeros([len(models), 5, 56, 56, CHANNEL_N], np.float32)
    cx, cy = 28, 28
    x[:, :, cy, cx, 3:] = 1.0
    for i in tqdm.trange(2000):
        if i == 200:
            x[:, 0, cy:] = x[:, 1, :cy] = 0
            x[:, 2, :, cx:] = x[:, 3, :, :cx] = 0
            x[:, 4, cy - 8:cy + 8, cx - 8:cx + 8] = 0
        vis = to_rgb(x)
        vis = np.vstack([np.hstack(row) for row in vis])
        vis = zoom(vis, 2)
        if (i < 400 and i % 2 == 0) or i % 8 == 0:
            vid.add(vis)
        if i == 200:
            for _ in range(29):
                vid.add(vis)
        for ca, row in zip(models, x):
            row[:] = ca(row)

mvp.ipython_display('regen2.mp4')

# ----------------------------------------------------- Planarian ------------------------------------------------------
# !wget -O planarian.zip 'https://github.com/google-research/self-organising-systems/blob/master/assets/growing_ca/planarian.zip?raw=true'
# !unzip -oq planarian.zip -d planarian

ca = CAModel()
ca.load_weights('planarian/train_log/8000')

x = np.zeros([1, 64, 96, CHANNEL_N], np.float32)
x[:, 32, 48, 3:] = 1.0
with VideoWriter('planarian.mp4', 30.0) as vid:
    for i in range(400):
        vid.add(zoom(to_rgb(x[0])))
        x = ca(x, angle=np.pi / 2.0)
        if i == 150:
            x = x.numpy()
            for k in range(24):
                x[:, :24] = np.roll(x[:, :24], 1, 2)
                x[:, -24:] = np.roll(x[:, -24:], -1, 2)
                vid.add(zoom(to_rgb(x[0])))
            for k in range(20):
                vid.add(zoom(to_rgb(x[0])))

mvp.ipython_display('planarian.mp4')
