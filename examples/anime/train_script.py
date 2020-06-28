import json
from pathlib import Path

import numpy as np

from core.image_utils import open_image
from core.petri_dish import PetriDish
from core.rule_model import UpdateRule
from core.rule_trainer import UpdateRuleTrainer

if __name__ == '__main__':
    main_root = Path(__file__).parent.absolute()

    # load experiment config
    experiment_config = str(main_root.joinpath('config.json'))
    with open(experiment_config, 'r') as conf:
        config = json.load(conf)

    # get target image
    target_path = str(main_root.joinpath('anime_girl.png'))
    target_img = open_image(target_path, max_size=40)

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

    # determine CA model type and experiments conditions
    exp_map = config['experiment_map']
    training_mode = config['experiment_type']
    use_pattern_pool = exp_map[training_mode]['use_pattern_pool']
    damage_n = exp_map[training_mode]['damage_n']

    # create trainer for UpdateRule object
    batch_size = config["train_config"]["batch_size"]
    train_steps = config["train_config"]["train_steps"]
    learning_rate = config["train_config"]["learning_rate"]
    boundaries = config["train_config"]["boundaries"]
    lr_multiplier = config["train_config"]["lr_multiplier"]
    grad_norm_value = config["train_config"]["grad_norm_value"]
    train_ca_step_range = tuple(config["train_config"]["train_ca_step_range"])

    trainer = UpdateRuleTrainer(root=main_root,
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
