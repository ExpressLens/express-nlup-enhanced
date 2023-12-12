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
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kyle Gorman",
    author_email="kylebgorman@gmail.com",
    url="http://github.com/cslu-nlp/nlup/",
    license="MIT",
    install_requires=["jsonpickle >= 0.9.0"],
    packages=["nlup"],
    keywords=[
        "nlp",
        "natural language processing",
        "text",
        "text processing",
        "ai",
        "artificial intelligence",
        "neural net",
        "perceptron",
        "data",
        "science",
        "statistics",
        "data science",
        "math",
        "machine learning",
        "computer science",
        "information theory",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Pytho