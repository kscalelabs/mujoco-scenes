"""Defines a function for generating different types of hfields."""

import argparse

import numpy as np
import PIL.Image


def get_rough_hfield(size: tuple[int, int]) -> PIL.Image.Image:
    """Returns a random rough field.

    Args:
        size: The size of the hfield.

    Returns:
        A PIL image of the hfield.
    """
    arr = np.random.randint(0, 255, size, dtype=np.uint8)
    return PIL.Image.fromarray(arr)


def get_steps_hfield(size: tuple[int, int], step_size: int) -> PIL.Image.Image:
    """Returns a field of steps.

    The steps are spaced randomly around the field.

    Args:
        size: The size of the hfield.
        step_size: The size of each step.

    Returns:
        A PIL image of the hfield.
    """
    w, h = size
    ws, hs = w // step_size, h // step_size
    arr = np.random.randint(0, 255, (hs, ws), dtype=np.uint8)
    arr = np.repeat(np.repeat(arr, step_size, axis=0), step_size, axis=1)
    wd, hd = w - ws * step_size, h - hs * step_size
    arr = np.pad(arr, ((0, hd), (0, wd)), mode="constant", constant_values=0)
    return PIL.Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["rough", "steps"])
    parser.add_argument("--size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--step-size", type=int, default=16)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.mode == "rough":
        img = get_rough_hfield(args.size)
    elif args.mode == "steps":
        img = get_steps_hfield(args.size, args.step_size)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    if args.output:
        img.save(args.output)
    else:
        img.show()


if __name__ == "__main__":
    main()
