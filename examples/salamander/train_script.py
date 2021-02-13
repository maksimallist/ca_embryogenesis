from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from core.experiment import ExperimentWatcher, TFKerasTrainer
from core.image_utils import load_emoji
from core.petri_dish import CADataGenerator, PetriDish
from core.rule_model import SimpleUpdateModel


@tf.function
def l2_loss(batch_x: np.array, batch_y: np.array):
    return tf.reduce_mean(tf.square(batch_x - batch_y), [-2, -3, -1])


if __name__ == '__main__':
    main_root = Path(__file__).parent.absolute()
    watcher = ExperimentWatcher(exp_name='salamander', root=main_root)

    # load target image
    target_img = load_emoji("🦎", max_size=40)
    target_padding = watcher.rlog(target_padding=8)
    target_padding_map = ((target_padding, target_padding), (target_padding, target_padding), (0, 0))
    padded_target = np.pad(target_img, target_padding_map, 'constant', constant_values=0.0)
    image_height, image_width = padded_target.shape[:2]

    watcher.log("cellar_automata", image_height=image_height, image_width=image_width)
    watcher.log_target(padded_target)

    ca = PetriDish(height=image_height,  # 56
                   width=image_width,  # 56
                   cell_states=watcher.rlog("cellar_automata", channel_n=16),
                   rgb_axis=watcher.rlog("cellar_automata", image_axis=(0, 1, 2)),
                   live_axis=watcher.rlog("cellar_automata", live_state_axis=3))
    ca.cell_state_initialization()

    watcher.log("training_process", loss_function='l2_loss')
    data_generator = CADataGenerator(ca_tensor=ca.cells_tensor,
                                     target=padded_target,
                                     set_size=watcher.rlog("training_process", "mode", pool_size=1024),
                                     damage_n=watcher.rlog("training_process", "mode", use_damage=3),
                                     reseed_batch=watcher.rlog("training_process", "mode", reseed_batch=True))

    model = SimpleUpdateModel(name="paper_model",
                              channels=watcher.rlog("neural_model", channel_n=16),
                              live_axis=watcher.rlog("neural_model", live_state_axis=3),
                              fire_rate=watcher.rlog("neural_model", cell_fire_rate=0.5),
                              life_threshold=watcher.rlog("neural_model", life_threshold=0.1),
                              perception_kernel_size=watcher.rlog("neural_model", perception_kernel_size=3),
                              perception_kernel_type=watcher.rlog("neural_model", perception_kernel_type='sobel'),
                              perception_kernel_norm_value=watcher.rlog("neural_model", perception_kernel_norm_value=8),
                              observation_angle=watcher.rlog("neural_model", observation_angle=0.0),
                              last_conv_filters=watcher.rlog("neural_model", last_conv_filters=128),
                              last_conv_kernel_size=watcher.rlog("neural_model", last_conv_kernel_size=1),
                              stochastic_update=watcher.rlog("neural_model", stochastic_update=True))

    lr = watcher.rlog("optimizer", learning_rate=2e-3)
    lr_multiplier = watcher.rlog("optimizer", lr_multiplier=0.1)
    boundaries_decay_values = [lr, lr * lr_multiplier]
    lr_scheduler = PiecewiseConstantDecay(boundaries=[watcher.rlog("optimizer", boundaries=2000)],
                                          values=boundaries_decay_values)
    model_optimizer = Adam(lr_scheduler)
    model.compile(optimizer=model_optimizer, loss=l2_loss)

    trainer = TFKerasTrainer(data_generator=data_generator,
                             model=model,
                             optimizer=model_optimizer,
                             watcher=watcher,
                             loss_function=l2_loss)

    trainer.train(train_steps=watcher.rlog("training_process", train_steps=2000),
                  batch_size=watcher.rlog("training_process", batch_size=8),
                  grad_norm_value=watcher.rlog("training_process", grad_norm_value=1e-8),
                  grow_steps=watcher.rlog("training_process", train_ca_step_range=(64, 96)))
