from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model

from core.image_utils import to_rgb


class ExperimentWatcher:
    config = {}
    date = datetime.now().strftime("%d.%m.%Y-%H.%M")

    checkpoints_folder = None
    pictures_folder = None
    video_folder = None
    last_pictures_folder = None
    tensorboard_logs = None

    def __init__(self, exp_name: str, root: Path):
        self.exp_root = root.joinpath(exp_name + '_' + self.date)
        self.log(exp_name=exp_name, exp_root=str(root))
        self._experiments_preparation()

    @staticmethod
    def _create_log_struct(struct: Dict, name: str) -> Dict:
        if name not in struct:
            struct[name] = {}
        return struct[name]

    def log(self, *args, **kwargs):
        if len(args) != 0:
            log_structure = self.config
            for arg in args:
                log_structure = self._create_log_struct(log_structure, arg)
                for att, val in kwargs.items():
                    log_structure[att] = val
        else:
            for att, val in kwargs.items():
                self.config[att] = val

    def rlog(self, *args, **kwargs):
        self.log(*args, **kwargs)
        values = tuple([v for v in kwargs.values()])
        if len(values) == 1:
            return values[0]
        elif len(values) > 1:
            return values
        else:
            return None

    def save_config(self):
        save_path = self.exp_root.joinpath('exp_config.json')
        with save_path.open('w') as outfile:
            json.dump(self.config, outfile, indent=4)

    def _experiments_preparation(self):
        self.exp_root.mkdir(parents=True, exist_ok=False)

        self.checkpoints_folder = self.exp_root.joinpath('checkpoints')
        self.checkpoints_folder.mkdir()

        self.pictures_folder = self.exp_root.joinpath('train_pictures')
        self.pictures_folder.mkdir()

        self.video_folder = self.exp_root.joinpath('train_video')
        self.video_folder.mkdir()

        self.tensorboard_logs = self.exp_root.joinpath(f"tb_logs")
        self.tensorboard_logs.mkdir()

        file_writer = tf.summary.create_file_writer(str(self.tensorboard_logs))
        file_writer.set_as_default()

    def log_target(self, target: np.array):
        target_image = to_rgb(target)
        # Using the file writer, log the target image.
        tf.summary.image("Target image", target_image[None, ...], step=0)
        # Save target np.array as jpeg image
        path = open(str(self.exp_root.joinpath('target_image.jpeg')), 'wb')
        target_image = np.uint8(np.clip(target_image, 0, 1) * 255)
        target_image = Image.fromarray(target_image)
        target_image.save(path, 'jpeg', quality=95)

    def _save_model(self, trainable_rule: Model, train_step: int):
        model_path = self.checkpoints_folder.joinpath("train_step_" + str(train_step))
        model_path.mkdir()
        trainable_rule.save(filepath=str(model_path), overwrite=True, save_format="tf")

    def _save_ca_state_as_image(self,
                                train_step: int,
                                post_state: np.array,
                                img_count: int = 8,
                                max_img_count: int = 25,
                                img_in_line: int = 4):
        path = open(str(self.pictures_folder) + f"/train_step_{train_step}.jpeg", 'wb')
        assert img_count <= max_img_count, ""
        assert len(post_state) >= img_count, ""

        images = []
        n_rows = img_count // img_in_line
        for i in range(1, n_rows, 1):
            images.append(np.hstack(to_rgb(post_state)[img_in_line * i:img_in_line * (i + 1)]))
        image = np.vstack(images)

        image = np.uint8(np.clip(image, 0, 1) * 255)
        image = Image.fromarray(image)
        image.save(path, 'jpeg', quality=95)

        tf.summary.image("Example of CA figures", to_rgb(post_state[0])[None, ...], step=train_step)

    def train_log(self, step, loss, trainable_rule, next_state_batch):
        if step % 10 == 0:
            tf.summary.scalar('loss_log', data=np.log10(loss), step=step)
            print(f"\r step: {step}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')

        if step % 100 == 0:
            self._save_ca_state_as_image(step, next_state_batch)

        if step % 1000 == 0:
            self._save_model(trainable_rule, step)


class TFKerasTrainer:
    def __init__(self,
                 data_generator,
                 model: Model,
                 optimizer,
                 watcher,
                 loss_function: Optional[Callable]):
        self.model = model  # compiled keras model
        self.watcher = watcher
        self.data_generator = data_generator

        self.optimizer = optimizer
        self.loss_function = loss_function

    def train_step(self, batch_size: int, grad_norm_value: float, grow_steps: Tuple[int, int]):
        iter_n = tf.random.uniform([], grow_steps[0], grow_steps[1], tf.int32)
        (batch_x, batch_ids), targets = self.data_generator.sample(batch_size)
        # batch_x: np.array; batch_ids: list indexes;

        with tf.GradientTape() as g:
            for _ in tf.range(iter_n):
                batch_x = self.model(batch_x)

            loss = tf.reduce_mean(self.loss_function(batch_x[..., :4], targets))  # batch_x to rgba

        grads = g.gradient(loss, self.model.weights)
        grads = [g / (tf.norm(g) + grad_norm_value) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.weights))

        # insert new ca tensors in pool
        self.data_generator.commit(batch_cells=batch_x, cells_idx=batch_ids)

        return loss, batch_x

    def train(self, train_steps: int, batch_size: int, grad_norm_value: float, grow_steps: Tuple[int, int]) -> None:
        print("[ Start training ... ]")
        self.watcher.save_config()
        for step in range(1, train_steps + 1, 1):
            loss, next_state_batch = self.train_step(batch_size, grad_norm_value, grow_steps)
            self.watcher.train_log(step, loss, self.model, next_state_batch)
