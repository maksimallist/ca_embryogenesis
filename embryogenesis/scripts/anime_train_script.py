import json
from pathlib import Path
from typing import Union

import PIL.Image
import PIL.ImageDraw
import numpy as np

from embryogenesis.petri_dish import PetriDish
from embryogenesis.rule_model import UpdateRule
from embryogenesis.rule_trainer import UpdateRuleTrainer


def open_image(path: Union[str, Path], max_size: int):
    img = PIL.Image.open(path)
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]

    return img


if __name__ == '__main__':
    main_root = Path("/home/mks/work/projects/cellar_automata_experiments")
    root = main_root.joinpath('experiments')

    # load experiment config
    experiment_config = str(main_root.joinpath('embryogenesis', 'scripts', 'anime_config.json'))
    with open(experiment_config, 'r') as conf:
        config = json.load(conf)

    # get target image
    target_path = str(main_root.joinpath('embryogenesis', 'data', 'clean_anime_target.png'))
    target_img = open_image(target_path, max_size=40)
    # image_height, image_width = target_img.shape[:2]
    # target_img = np.concatenate([target_img, np.zeros((image_height, image_width, 1))], axis=2)

    # determine CA model type and experiments conditions
    exp_map = config['experiment_map']
    training_mode = config['experiment_type']
    use_pattern_pool = exp_map[training_mode]['use_pattern_pool']
    damage_n = exp_map[training_mode]['damage_n']

    # pad target image to
    target_padding = config['target_padding']
    target_padding_map = ((target_padding, target_padding), (target_padding, target_padding), (0, 0))
    padded_target = np.pad(target_img, target_padding_map, 'constant', constant_values=0.0)
    image_height, image_width = padded_target.shape[:2]

    # create object that will creates CA cells grids
    sampler = PetriDish(height=image_height,
                        width=image_width,
                        channel_n=config['ca_params']['channel_n'],
                        pool_size=config['ca_params']['pool_size'],
                        live_state_axis=config['ca_params']['live_state_axis'],
                        image_axis=tuple(config['ca_params']['image_axis']))

    # create network that determine CA update rule
    model = UpdateRule(name='anime',
                       channel_n=config['ca_params']['channel_n'],
                       fire_rate=config['update_rule']['cell_fire_rate'],
                       life_threshold=config['update_rule']['life_threshold'],
                       conv_1_filters=config['update_rule']['conv_1_filters'],
                       conv_kernel_size=config['update_rule']['conv_kernel_size'],
                       step_size=config['update_rule']['step_size'])

    # create trainer for UpdateRule object
    batch_size = config["train_config"]["batch_size"]
    train_steps = config["train_config"]["train_steps"]
    learning_rate = config["train_config"]["learning_rate"]
    boundaries = config["train_config"]["boundaries"]
    lr_multiplier = config["train_config"]["lr_multiplier"]
    grad_norm_value = config["train_config"]["grad_norm_value"]
    train_ca_step_range = tuple(config["train_config"]["train_ca_step_range"])

    trainer = UpdateRuleTrainer(root=root,
                                exp_name='anime_regen',
                                petri_dish=sampler,
                                rule_model=model,
                                target_image=padded_target,
                                use_pattern_pool=use_pattern_pool,
                                damage_n=damage_n,
                                batch_size=batch_size,
                                train_steps=train_steps,
                                learning_rate=learning_rate,
                                boundaries=boundaries,
                                lr_multiplier=lr_multiplier,
                                train_ca_step_range=train_ca_step_range,
                                grad_norm_value=grad_norm_value)

    # run training
    trainer.train()
