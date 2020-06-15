import os
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
main_root = Path("/home/mks/work/projects/cellar_automata_experiments")
checkpoints = main_root.joinpath('experiments', 'test_20200615-233950', 'checkpoints', '1000')
video_save_path = main_root.joinpath('experiments', 'test_20200615-233950')

model = load_model(checkpoints)
test_input = np.random.random((1, 72, 72, 16))
test_target = np.random.random((1, 72, 72, 16))

a = model(test_input)
z = model.predict(test_input)
