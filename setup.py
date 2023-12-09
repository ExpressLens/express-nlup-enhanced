from setuptools import setup
from os import path

description = ("Core libraries for natural language processing",)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="nlup",
    include_package_data=True,
    version="0.9",
    description=description,
    long_descr