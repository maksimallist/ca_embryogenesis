from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

from core.image_utils import to_rgb
from core.petri_dish import PetriDish
from core.video_writer import VideoWriter, tile2d, zoom


class MorphCA:
    """
    The class of reproducing functional of a cellular automaton. That is, the class contains both a set of cells and
    their states, and cell renewal rules. Also, for this class, methods are implemented that allow you to apply update
    rules to a set of cells to obtain a new set of their states, thereby simulating the process of developing
    the entire system for a specified number of steps. One step in the simulation is the one-time application of the
    update rule to the cell set, and the replacement of the old set of cell states with a new one.
    """

    def __init__(self,
                 petri_dish: PetriDish,
                 update_model: tf.keras.Model,
                 print_summary: bool = True,
                 compatibility_test: bool = True):
        """
        Loading the tensorflow checkpoints with neural network that determine cellar automaton update rule. And create
        instance if class PetriDish that determine functional of cells set and their states.
        """
        self.petri_dish = petri_dish
        self.update_model = update_model
        if print_summary:
            self.print_summary()
        if compatibility_test:
            self.compatibility_test()

    def print_summary(self) -> None:
        self.petri_dish.summary()
        print()
        self.update_model.summary()

    def compatibility_test(self) -> None:
        try:
            zeros = np.zeros_like(self.petri_dish.cells_tensor)
            _ = self.update_model(zeros)
            print(f"Compatibility test pass.")
        finally:
            raise ValueError("The shapes of the Petri dish and the updated model do not match.")

    def step(self) -> None:
        """ Once applies the update rule to a set of cells states. """
        ca_tensor = self.petri_dish.cells_tensor
        self.petri_dish.cells_tensor = self.update_model(ca_tensor)

    def run_growth_simulation(self,
                              steps: int,
                              return_final_state: bool = False,
                              write_video: bool = True,
                              save_video_path: Optional[Path] = None,
                              video_name: Optional[str] = None):
        """
        Run simulations of growth of cellular automata.

        Args:
            steps: number of simulation steps
            return_final_state: trigger that determine return final cell states or not
            write_video:
            save_video_path:
            video_name:

        Returns:
            None or cells state
        """
        if write_video:
            if video_name is None:
                save_video_path = save_video_path.joinpath('mca_grow.mp4')
            else:
                video_name += '.mp4'
                save_video_path = save_video_path.joinpath(video_name)

            video_writer = VideoWriter(str(save_video_path))

            # run grow
            with video_writer as video:
                video.add(zoom(tile2d(to_rgb(self.petri_dish.cells_tensor), 5), 2))
                for _ in tqdm.trange(steps):
                    self.step()
                    video.add(zoom(tile2d(to_rgb(self.petri_dish.cells_tensor), 5), 2))
        else:
            for _ in tqdm.trange(steps):
                self.step()

        if return_final_state:
            return self.petri_dish.cells_tensor

    def save_state(self, save_path: Path, image_name: str):
        rgb_image = Image.fromarray(zoom(tile2d(to_rgb(self.petri_dish.cells_tensor), 5), 2))
        rgb_image.save(save_path.joinpath(image_name + ".jpeg"))
