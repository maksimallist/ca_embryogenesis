import json
from pathlib import Path

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from core.experiment import l2_loss, ExperimentWatcher, TFKerasTrainer
from core.image_utils import load_emoji
from core.petri_dish import CADataGenerator, PetriDish
from core.rule_model import SimpleUpdateModel

if __name__ == '__main__':
    main_root = Path(__file__).parent.absolute()

    # load experiment config
    experiment_config = str(main_root.joinpath('config.json'))
    with open(experiment_config, 'r') as conf:
        config = json.load(conf)

    target_img = load_emoji("🦎", max_size=40)

    # pad target image to
    target_padding = config['target_padding']
    target_padding_map = ((target_padding, target_padding), (target_padding, target_padding), (0, 0))
    padded_target = np.pad(target_img, target_padding_map, 'constant', constant_values=0.0)
    image_height, image_width = padded_target.shape[:2]

    ca = PetriDish(height=image_height,
                   width=image_width,
                   cell_states=config['channel_n'],  # 16
                   rgb_axis=config['image_axis'],  # (0, 1, 2),
                   live_axis=config['live_state_axis'])  # 3

    ca.cell_state_initialization()

    data_generator = CADataGenerator(ca_tensor=ca.cells_tensor,
                                     target=padded_target,
                                     set_size=config['pool_size'],  # 1024
                                     damage_n=config['damage_n'],  # 3
                                     reseed_batch=config['reseed_batch'])  # True

    model = SimpleUpdateModel(name="paper_model",
                              channels=config['channel_n'],  # 16
                              life_threshold=config['life_threshold'],  # 0.1
                              live_axis=config['live_state_axis'],  # 3
                              perception_kernel_size=config['perception_kernel_size'],  # 3
                              perception_kernel_type=config['perception_kernel_type'],  # 'sobel'
                              perception_kernel_norm_value=config['perception_kernel_norm_value'],  # 8
                              observation_angle=config['observation_angle'],  # 0.0
                              last_conv_filters=config['last_conv_filters'],  # 128
                              last_conv_kernel_size=config['last_conv_kernel_size'],  # 1
                              stochastic_update=config['stochastic_update'],  # True
                              fire_rate=config['cell_fire_rate'])  # 0.5

    boundaries_decay_values = [config['learning_rate'], config['learning_rate'] * config['lr_multiplier']]
    lr_scheduler = PiecewiseConstantDecay(boundaries=[config['boundaries']],
                                          values=boundaries_decay_values)
    model_optimizer = Adam(lr_scheduler)
    model.compile(optimizer=model_optimizer, loss=l2_loss)

    watcher = ExperimentWatcher(root=main_root,
                                exp_name=config['exp_name'],
                                target=padded_target)

    trainer = TFKerasTrainer(data_generator=data_generator,
                             model=model,
                             optimizer=model_optimizer,
                             watcher=watcher,
                             loss_function=l2_loss)

    trainer.train(train_steps=config['train_steps'],
                  grad_norm_value=config['grad_norm_value'],
                  grow_steps=tuple(config['train_ca_step_range']))
