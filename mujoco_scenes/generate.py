#!/usr/bin/env python
"""This script can be used to generate new grids."""

import argparse

import numpy as np
from PIL import Image


def generate_patch_texture(patch_type: str, size: int) -> Image.Image:
    match patch_type:
        case "rough":
            # Rough: random noise.
            arr = (np.random.rand(size, size, 3) - 0.5) * 0.03 + 0.5

        case "smooth":
            # Smooth: uniform gray texture (constant value).
            arr = np.ones((size, size, 3)) * 0.5

        case "hill":
            # Hill: pyramid shape that rises from 0.5 at edges to 1.0 at center
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xs, ys = np.meshgrid(x, y)
            # Create pyramid using manhattan distance
            r = np.maximum(np.abs(xs), np.abs(ys))
            # Scale from 0.5 to 1.0
            intensity = np.clip(1.0 - r, 0, 1) * 0.5 + 0.5
            arr = np.stack([intensity] * 3, axis=-1)

        case "valley":
            # Valley: pyramid shape that drops from 0.5 at edges to 0.0 at center
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xs, ys = np.meshgrid(x, y)
            # Create pyramid using manhattan distance
            r = np.maximum(np.abs(xs), np.abs(ys))
            # Scale from 0.5 to 0.0
            intensity = np.clip(r, 0, 1) * 0.5
            arr = np.stack([intensity] * 3, axis=-1)

        case _:
            raise ValueError(f"Invalid patch type: {patch_type}")

    # Convert from float values in [0, 1] to uint8 [0, 255]
    arr_uint8 = (arr * 255).astype("uint8")
    return Image.fromarray(arr_uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image grid of floor patches with various textures.")
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of rows in the grid (default: 4)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the grid (default: 4)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Pixel size (width and height) of each patch (default: 256)",
    )
    parser.add_argument(
        "--freq-rough",
        type=float,
        default=1.0,
        help="Weight for 'rough' patch (default: 1.0)",
    )
    parser.add_argument(
        "--freq-smooth",
        type=float,
        default=1.0,
        help="Weight for 'smooth' patch (default: 1.0)",
    )
    parser.add_argument(
        "--freq-hill",
        type=float,
        default=1.0,
        help="Weight for 'hill' patch (default: 1.0)",
    )
    parser.add_argument(
        "--freq-valley",
        type=float,
        default=1.0,
        help="Weight for 'valley' patch (default: 1.0)",
    )
    # Output file.
    parser.add_argument(
        "--output",
        type=str,
        default="patch_hfield.png",
        help="Output image file (default: generated_grid.png)",
    )
    args = parser.parse_args()

    # Define patch types in specific order
    patch_types = ["hill", "valley", "smooth", "rough"]
    frequencies = [args.freq_hill, args.freq_valley, args.freq_smooth, args.freq_rough]

    # Calculate number of rows for each type based on frequencies
    total_freq = sum(frequencies)
    rows_per_type = [int(round(f * args.rows / total_freq)) for f in frequencies]
    # Adjust to ensure total rows matches args.rows
    while sum(rows_per_type) < args.rows:
        rows_per_type[np.argmax(frequencies)] += 1
    while sum(rows_per_type) > args.rows:
        rows_per_type[np.argmin(frequencies)] -= 1

    # Generate patch images programmatically
    patch_images = {}
    for p_type in patch_types:
        patch_images[p_type] = generate_patch_texture(p_type, args.patch_size)

    # Create a new blank image for the grid
    grid_width = args.cols * args.patch_size
    grid_height = args.rows * args.patch_size
    grid_image = Image.new("RGB", (grid_width, grid_height))

    current_row = 0
    for patch_type, num_rows in zip(patch_types, rows_per_type):
        patch_img = patch_images[patch_type]
        for row in range(num_rows):
            for j in range(args.cols):
                x = j * args.patch_size
                y = (current_row + row) * args.patch_size
                grid_image.paste(patch_img, (x, y))
        current_row += num_rows

    grid_image.save(args.output)
    print(f"Generated grid image saved to {args.output}")


if __name__ == "__main__":
    # python -m mujoco_scenes.generate
    main()
