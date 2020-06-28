from pathlib import Path

from core.cellar_automata import MorphCA

main_root = Path(__file__).parent.absolute()
checkpoints = main_root.joinpath('... checkpoint_name ...', 'checkpoints', '... number of checkpoint ...')

cellar_automata = MorphCA(rule_model_path=checkpoints,
                          write_video=True,
                          video_name=' ... ',
                          save_video_path=main_root)

cellar_automata.run_growth(steps=450, return_state=False)
