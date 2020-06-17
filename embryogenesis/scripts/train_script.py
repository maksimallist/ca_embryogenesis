import json
from pathlib import Path

import numpy as np

from embryogenesis.petri_dish import PetriDish
from embryogenesis.rule_model import UpdateRule
from embryogenesis.rule_trainer import UpdateRuleTrainer
from embryogenesis.utils import load_image

if __name__ == '__main__':
    main_root = Path("/home/mks/work/projects/cellar_automata_experiments")
    root = main_root.joinpath('experiments')

    # load experiment config
    experiment_config = str(main_root.joinpath('embryogenesis', 'scripts', 'exp_config.json'))
    with open(experiment_config, 'r') as conf:
        config = json.load(conf)

    # get target image
    # source = "https://github.com/google-research/self-organising-systems/blob/master/assets/"
    # planaria = source + "growing_ca/planaria2_48.png?raw=true"
    salamander = "https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u1f98e.png"
    target_img = load_image(salamander, max_size=40)

    # determine CA model type and experiments conditions
    # "experiment_map": {
    #     "Growing": 0,
    #     "Persistent": 1,
    #     "Regenerating": 2
    # },

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
    model = UpdateRule(name='salamander_2',
                       channel_n=config['ca_params']['channel_n'],
                       fire_rate=config['update_rule']['cell_fire_rate'],
                       life_threshold=config['update_rule']['life_threshold'],
                       conv_1_filters=config['update_rule']['conv_1_filters'],
                       conv_kernel_size=config['update_rule']['conv_kernel_size'],
                       step_size=config['update_rule']['step_size'])

    # create trainer for UpdateRule object
    trainer = UpdateRuleTrainer(root=root,
                                exp_name='salamander_2_1000',
                                petri_dish=sampler,
                                rule_model=model,
                                target_image=padded_target,
                                use_pattern_pool=use_pattern_pool,
                                damage_n=damage_n,
                                **config['experiment_config'])

    # run training
    trainer.train()
