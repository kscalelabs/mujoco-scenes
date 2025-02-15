"""Defines the CLI for rendering Mujoco scenes."""

import argparse

from mujoco_scenes.mjcf import load_mjmodel, list_scenes
import mujoco.viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_path", type=str, help="Path to the robot URDF file.")
    parser.add_argument("scene", choices=list_scenes(), help="The scene to render.")
    args = parser.parse_args()

    model = load_mjmodel(args.robot_path, args.scene)

    mujoco.viewer.launch(model)


if __name__ == "__main__":
    # python -m mujoco_scenes.cli
    main()
