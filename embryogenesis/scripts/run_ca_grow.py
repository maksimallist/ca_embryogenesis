import os
from pathlib import Path

from embryogenesis.cellar_automata import MorphCA

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
main_root = Path("/home/mks/work/projects/cellar_automata_experiments")
checkpoints = main_root.joinpath('experiments', 'test_20200616-012418', 'checkpoints', '200')
video_save_path = main_root.joinpath('experiments', 'test_20200616-012418')

cellar_automata = MorphCA(rule_model_path=checkpoints,
                          write_video=True,
                          video_name='test',
                          save_video_path=video_save_path)

cellar_automata.run_growth(steps=1000, return_state=False)
