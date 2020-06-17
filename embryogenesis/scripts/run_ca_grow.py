import os
from pathlib import Path

from embryogenesis.cellar_automata import MorphCA

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
main_root = Path("/home/mks/work/projects/cellar_automata_experiments")
checkpoints = main_root.joinpath('experiments', 'salamander_2_5000_20200617-225612', 'checkpoints', '1000')
video_save_path = main_root.joinpath('experiments', 'salamander_2_5000_20200617-225612')

cellar_automata = MorphCA(rule_model_path=checkpoints,
                          write_video=True,
                          video_name='salamander_2_mini',
                          save_video_path=video_save_path)

cellar_automata.run_growth(steps=150, return_state=False)
