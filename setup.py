# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import glob
import re

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()

requirements = [
    "etils",
    "mujoco",
    "numpy",
    "Pillow",
]

requirements_dev = [
    "black",
    "darglint",
    "mypy",
    "pytest",
    "ruff",
]

# Collects package data.
package_data = ["mujoco_scenes/py.typed"]
for ext in (".xml", ".png"):
    package_data.extend(glob.glob(f"mujoco_scenes/**/*.{ext}", recursive=True))

with open("mujoco_scenes/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in mujoco_scenes/__init__.py"
version: str = version_re.group(1)


setup(
    name="mujoco-scenes",
    version=version,
    description="The mujoco-scenes project",
    author="Benjamin Bolte",
    url="https://github.com/kscalelabs/mujoco-scenes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={"mujoco_scenes": package_data},
    entry_points={
        "console_scripts": [
            "mujoco-scene = mujoco_scenes.cli:main",
        ],
    },
)
