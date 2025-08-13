# mujoco-scenes

This package contains a collection of utility functions for generating different types of scenes in Mujoco.

This is a programmatic way to compose scene files, inspired by [Mujoco Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## Usage

To load a robot into a particular scene, you can use the following code:

```python
from mujoco_scenes.mjcf import load_mjmodel

model = load_mjmodel("path/to/robot.xml", "smooth")

# To view the model, you can use:
mujoco.viewer.launch(model)
```


### Command to create terrain `patch_light`
```bash
python generate.py --freq-hill 0 --freq-valley 0 --patch-size 64  --rows 5 --freq-hill-valley 2 --freq-rough 2  --rough-amplitude 0.5 --hill-amplitude 0.5 --valley-amplitude 0.5 --output templates/assets/patch_light.png
```