import os
from setuptools import setup
import pathlib


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


def read_requirements_file(filename):
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(file_path) as f:
        return [line.strip() for line in f]


setup(
    name="rl_x",
    description="RL-X: A framework for Reinforcement Learning research",
    long_description=long_description,
    url="https://github.com/nico-bohlinger/RL-X",
    author="Nico Bohlinger",
    author_email="nico.bohlinger@gmail.com",
    version="0.0.1",
    packages=["rl_x"],
    install_requires=read_requirements_file("requirements.txt"),
    license="MIT",
)
