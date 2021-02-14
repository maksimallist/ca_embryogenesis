import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model

from core.cellar_automata import MorphCA
from core.image_utils import to_rgb
from core.cell_cultures import PetriDish


def is_json_serializable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class Watcher:
    config = {}

    def __init__(self, *args, **kwargs):
        self.log(*args, **kwargs)

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
                    if is_json_serializable(val):
                        log_structure[att] = val
                    setattr(self, att, val)
        else:
            for att, val in kwargs.items():
                if is_json_serializable(val):
                    self.config[att] = val
                setattr(self, att, val)

    def rlog(self, *args, **kwargs):
        self.log(*args, **kwargs)
        values = tuple([v for v in kwargs.values()])
        if len(values) == 1:
            return values[0]
        elif len(values) > 1:
            return values
        else:
            return None

    def save_conf(self, save_path: Path):
        save_path = save_path.joinpath('exp_config.json')
        with save_path.open('w') as outfile:
            json.dump(self.config, outfile, indent=4)


class ExpWatcher(Watcher):
    _checkpoints_folder = None
    _pictures_folder = None
    _video_folder = None
    _tensorboard_logs = None

    def __init__(self, exp_name: str, root: Path, *args, **kwargs):
        super(ExpWatcher, self).__init__(exp_name=exp_name, root=str(root), *args, **kwargs)
        date = datetime.now().strftime("%d.%m.%Y-%H.%M")
        self.log(exp_date=date)  # exp_name=exp_name, root=str(root)
        self.exp_root = root.joinpath(exp_name + '_' + date)
        self._experiments_preparation()

    def _experiments_preparation(self):
        self.exp_root.mkdir(parents=True, exist_ok=False)

        self._checkpoints_folder = self.exp_root.joinpath('checkpoints')
        self._checkpoints_folder.mkdir()

        self._pictures_folder = self.exp_root.joinpath('train_pictures')
        self._pictures_folder.mkdir()

        self._video_folder = self.exp_root.joinpath('train_video')
        self._video_folder.mkdir()

        self._tensorboard_logs = self.exp_root.joinpath(f"tb_logs")
        self._tensorboard_logs.mkdir()

        file_writer = tf.summary.create_file_writer(str(self._tensorboard_logs))
        file_writer.set_as_default()

    def _save_model(self, trainable_rule: Model, train_step: int):
        model_path = self._checkpoints_folder.joinpath("train_step_" + str(train_step))
        model_path.mkdir()
        trainable_rule.save(filepath=str(model_path), overwrite=True, save_format="tf")

    def _save_ca_state_as_image(self,
                                train_step: int,
                                post_state: np.array,
                                img_count: int = 8,
                                max_img_count: int = 25,
                                img_in_line: int = 4):
        path = open(str(self._pictures_folder) + f"/train_step_{train_step}.jpeg", 'wb')
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

    def _save_ca_video(self, train_step: int, trainable_rule: Model):
        print(f"[\n Saving a video recording of the growth of a cellular automaton ... ]")
        print(f"[ Petri dish creation started ]")

        # todo: придумать как от этого избавиться
        image_height = getattr(self, "image_height", None)
        image_width = getattr(self, "image_width", None)
        channel_n = getattr(self, "channel_n", None)
        image_axis = getattr(self, "image_axis", None)
        live_state_axis = getattr(self, "live_state_axis", None)

        petri_dish = PetriDish(height=image_height,
                               width=image_width,
                               cell_states=channel_n,
                               rgb_axis=image_axis,
                               live_axis=live_state_axis,
                               print_summary=False)
        petri_dish.cell_state_initialization()
        print(f"[ Petri dish creation completed ]")

        # create cellar automata for embryogenesis
        cellar_automata = MorphCA(petri_dish=petri_dish,
                                  update_model=trainable_rule,
                                  print_summary=False,
                                  compatibility_test=False)
        print(f"[ The simulation of the growth of the cellular automaton was launched ]")
        cellar_automata.run_growth_simulation(steps=200,
                                              return_final_state=False,
                                              write_video=True,
                                              save_video_path=self._video_folder,
                                              video_name=f"train_step_{train_step}")
        print(f"[ The video recording of the cellular automaton growth is now complete. ]")

    def save_config(self):
        super(ExpWatcher, self).save_conf(self.exp_root)

    def log_target(self, target: np.array):
        target_image = to_rgb(target)
        # Using the file writer, log the target image.
        tf.summary.image("Target image", target_image[None, ...], step=0)
        # Save target np.array as jpeg image
        path = open(str(self.exp_root.joinpath('target_image.jpeg')), 'wb')
        target_image = np.uint8(np.clip(target_image, 0, 1) * 255)
        target_image = Image.fromarray(target_image)
        target_image.save(path, 'jpeg', quality=95)

    def log_train(self, step, loss, trainable_rule, next_state_batch):
        if step == 1:
            tf.summary.scalar('loss_log', data=np.log10(loss), step=step)
            print(f"\r step: {step}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')
            self._save_ca_state_as_image(step, next_state_batch)
            self._save_ca_video(step, trainable_rule)
            self._save_model(trainable_rule, step)

        if step % 10 == 0:
            tf.summary.scalar('loss_log', data=np.log10(loss), step=step)
            print(f"\r step: {step}, log10(loss): {np.round(np.log10(loss), decimals=3)}", end='')

        if step % 100 == 0:
            self._save_ca_state_as_image(step, next_state_batch)

        if step % 1000 == 0:
            self._save_model(trainable_rule, step)
            self._save_ca_video(step, trainable_rule)
