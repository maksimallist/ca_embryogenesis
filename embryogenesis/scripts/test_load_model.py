import os
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
main_root = Path("/home/mks/work/projects/cellar_automata_experiments")
checkpoints = main_root.joinpath('experiments', 'test_20200616-012418', 'checkpoints', '200')
video_save_path = main_root.joinpath('experiments', 'test_20200616-012418')

test_input = np.random.random((1, 72, 72, 16))
test_target = np.random.random((1, 72, 72, 16))

model = load_model(checkpoints)

a = model(test_input)
z = model.predict(test_input)
