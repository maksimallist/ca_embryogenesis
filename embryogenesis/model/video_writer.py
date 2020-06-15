import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


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

# ---------------------------------------------------------------------------------------------------------------------
# with VideoWriter('teaser.mp4') as vid:
#     x = np.zeros([len(EMOJI), 64, 64, CHANNEL_N], np.float32)
#
#     # grow
#     for i in tqdm.trange(200):
#         k = i // 20
#
#         if i % 20 == 0 and k < len(EMOJI):
#             x[k, 32, 32, 3:] = 1.0
#
#         vid.add(zoom(tile2d(to_rgb(x), 5), 2))
#
#         for ca, xk in zip(models, x):
#             xk[:] = ca(xk[None, ...])[0]
#
#     # damage
#     mask = PIL.Image.new('L', (64 * 5, 64 * 2))
#     draw = PIL.ImageDraw.Draw(mask)
#
#     for i in tqdm.trange(400):
#         cx, r = i * 3 - 20, 6
#         y1, y2 = 32 + np.sin(i / 5 + np.pi) * 8, 32 + 64 + np.sin(i / 5) * 8
#         draw.rectangle((0, 0, 64 * 5, 64 * 2), fill=0)
#         draw.ellipse((cx - r, y1 - r, cx + r, y1 + r), fill=255)
#         draw.ellipse((cx - r, y2 - r, cx + r, y2 + r), fill=255)
#         x *= 1.0 - (np.float32(mask).reshape(2, 64, 5, 64)
#                     .transpose([0, 2, 1, 3]).reshape(10, 64, 64, 1)) / 255.0
#         if i < 200 or i % 2 == 0:
#             vid.add(zoom(tile2d(to_rgb(x), 5), 2))
#
#         for ca, xk in zip(models, x):
#             xk[:] = ca(xk[None, ...])[0]
#
#     # fade out
#     last = zoom(tile2d(to_rgb(x), 5), 2)
#     for t in np.linspace(0, 1, 30):
#         vid.add(last * (1.0 - t) + t)
#
# mvp.ipython_display('teaser.mp4', loop=True)
