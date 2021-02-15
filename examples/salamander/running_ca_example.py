import json
from pathlib import Path

from tensorflow.keras.models import load_model

from nca.cellar_automata import MorphCA
from nca.cell_cultures import PetriDish

main_root = Path(__file__).parent.absolute()

# load experiment config
experiment_config = str(main_root.joinpath('... exp name ...', 'exp_config.json'))
with open(experiment_config, 'r') as conf:
    config = json.load(conf)

# initialize Petri dish
petri_dish = PetriDish(height=config["cellar_automata"]['image_height'],
                       width=config["cellar_automata"]['image_width'],
                       cell_states=config["cellar_automata"]['channel_n'],
                       rgb_axis=config["cellar_automata"]['image_axis'],
                       live_axis=config["cellar_automata"]['live_state_axis'])
petri_dish.cell_state_initialization()

# load trained neural network that is implement update rule for cells
checkpoints = main_root.joinpath('... exp name ...', 'checkpoints', 'train_step_ ...')
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
                                      video_name="salamander_growth")
