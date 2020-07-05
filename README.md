# An attempt to learn how to design complex self-organizing systems.
At this stage, a scientific article https://distill.pub/2020/growing-ca/ is being 
reproduced in this repository. The original notebook has been replaced with a more 
convenient code platform for experimentation. The results of the original article 
were reproduced, and my own experiments were added.

## Instruction for starting experiments.
For reproducing results of original work, watch the ./examples/salamander/config.json file. 
It is contains all experiments parameters such as "experiment_type", "batch_size", 
"train_steps", "learning_rate" and e.t.c To start training, run the train_script.py file.
A folder will be created in the local folder where checkpoints and tensorboard logs will be stored.

After the training is over, you can select the checkpoint and run a simulation of the 
growth of the cellular automaton, by running the run_ca_grow.py script. As a result in the local 
folder will be created video file with cellular automaton growth (extension of video file *.mp4).
Parameters such as the number of simulation steps, the save path for the video file, the video file 
name and the number of control points can be defined in the run_ca_grow.py script.

You can also select any of the images presented in the article as a target for training by 
simply selecting the desired character from the ./examples/salamander/constant.py file and 
substituting it into the train script.