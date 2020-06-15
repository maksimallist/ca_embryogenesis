from typing import Optional, Union
from pathlib import Path

from tensorflow.keras.models import load_model
from embryogenesis.model.petri_dish import PetriDish


class MorphCA:
    def __init__(self,
                 rule_model_path: Union[str, Path],
                 write_video: bool = False,
                 save_video_path: Optional[Union[str, Path]] = None):
        self.write_video = write_video
        self.save_video_path = save_video_path
        assert (self.save_video_path and self.write_video) is True, ValueError("")

        self.rule = load_model(rule_model_path)
        self.petri_dish = PetriDish(target_image=, target_padding=, channel_n=)




