import os
from pathlib import Path

from core.cellar_automata import MorphCA

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
main_root = Path(" ... ")
checkpoints = main_root.joinpath('experiments', '', 'checkpoints', '')
video_save_path = main_root.joinpath('experiments', '')

cellar_automata = MorphCA(rule_model_path=checkpoints,
                          write_video=True,
                          video_name=' ... ',
                          save_video_path=video_save_path)

cellar_automata.run_growth(steps=450, return_state=False)
