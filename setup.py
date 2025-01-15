import os
import sys
from setuptools import setup

# Handle version import without importing the full package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym_cellular_automata"))
from version import VERSION

# For TPU support: pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
setup(
    name="gym_cellular_automata",
    packages=["gym_cellular_automata"],
    version=VERSION,
    description="Cellular Automata Environments for Reinforcement Learning",
    url="https://github.com/elbecerrasoto/gym-cellular-automata",
    author="Emanuel Becerra Soto",
    author_email="elbecerrasoto@gmail.com",
    license="MIT",
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
        "scipy",
        "svgpath2mpl",
        "jax",
        "jaxlib",
        "moviepy",
        "gif",
        "flax",
        "torch",
        "tensorboard",
        "wandb",
    ],
    tests_require=["pytest", "pytest-cov", "pytest-repeat", "pytest-randomly"],
    python_requires=">=3.9",
)
