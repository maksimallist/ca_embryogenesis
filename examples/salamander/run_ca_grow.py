import json
from pathlib import Path

from tensorflow.keras.models import load_model

from core.cellar_automata import MorphCA
from core.petri_dish import PetriDish

main_root = Path(__file__).parent.absolute()

# load experiment config
experiment_config = str(main_root.joinpath('config.json'))
with open(experiment_config, 'r') as conf:
    config = json.load(conf)

# initialize Petri dish
petri_dish = PetriDish(height=56,
                       width=56,
                       cell_states=config['channel_n'],  # 16
                       rgb_axis=config['image_axis'],  # (0, 1, 2),
                       live_axis=config['live_state_axis'])  # 3
petri_dish.cell_state_initialization()

# load trained neural network that is implement update rule for cells
checkpoints = main_root.joinpath('salamander_example_12.02.2021-11.35', 'checkpoints', 'train_step_7000')
update_model = load_model(str(checkpoints), compile=False)
update_model.summary()

# create cellar automata for embryogenesis
cellar_automata = MorphCA(petri_dish=petri_dish,
                          update_model=update_model,
                          print_summary=True,
                          compatibility_test=True)

cellar_automata.run_growth_simulation(steps=450,
                                      return_final_state=False,
                                      write_video=True,
                                      save_video_path=main_root,
                                      video_name="salamander_grow")
