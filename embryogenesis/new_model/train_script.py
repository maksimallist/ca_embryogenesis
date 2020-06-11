import json
from pathlib import Path

from embryogenesis.new_model.petri_dish import PetriDish
from embryogenesis.new_model.update_rule_model import UpdateRule
from embryogenesis.new_model.update_rule_trainer import UpdateRuleTrainer
from embryogenesis.new_model.utils import load_image

main_root = Path("/home/mks/work/projects/cellar_automata_experiments")
root = main_root.joinpath('experiments')

# load experiment config
experiment_config = str(main_root.joinpath('embryogenesis', 'new_model', 'exp_config.json'))
with open(experiment_config, 'r') as conf:
    config = json.load(conf)


def load_emoji(max_size):
    url = f"https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u1f98e.png"
    return load_image(url, max_size)


target_img = load_emoji(max_size=40)

sampler = PetriDish(target_image=target_img,
                    target_padding=config['ca_params']['target_padding'],
                    morph_axis=tuple(config['ca_params']['morph_axis']),
                    channel_n=config['ca_params']['channel_n'],
                    pool_size=config['ca_params']['pool_size'],
                    live_state_axis=config['ca_params']['live_state_axis'])

model = UpdateRule(name='test_model',
                   channel_n=config['ca_params']['channel_n'],
                   fire_rate=config['update_rule']['cell_fire_rate'],
                   life_threshold=config['update_rule']['life_threshold'],
                   conv_1_filters=config['update_rule']['conv_1_filters'],
                   conv_kernel_size=config['update_rule']['conv_kernel_size'],
                   step_size=config['update_rule']['step_size'])

exp_map = config['experiment_map']
EXPERIMENT_N = exp_map[config['experiment_type']]

use_pattern_pool = [0, 1, 1][EXPERIMENT_N]
damage_n = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

padded_target = sampler.return_target()

trainer = UpdateRuleTrainer(root=root,
                            exp_name='test',
                            petri_dish=sampler,
                            rule_model=model,
                            target_image=padded_target,
                            use_pattern_pool=use_pattern_pool,
                            damage_n=damage_n,
                            **config['experiment_config'])

trainer.train()
