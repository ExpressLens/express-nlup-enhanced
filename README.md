`express-nlup-enhanced` is an updated version of base libraries used extensively in natural language processing projects. Check out the following features:

* `confusion.py`: This file contains objects for classifier evaluation
* `decorators.py`: It has clever decorators for various functionalities
* `jsonable.py`: This file contains a mix-in that allows the state of most objects to be serialized to and deserialized from compressed JSON
* `perceptron.py`: It contains perceptron-like classifiers (binary and multiclass), including some strategies for structured prediction
* `reader.py`: Comprises classes and readers for tagged and dependency-parsed data
* `timer.py`: A `with`-block that records the wall clock time elapsed

These libraries have been tested on CPython 3.4.1 and PyPy 3.2.5 (PyPy version 2.3.1). It is to be noted that Python 2 is not supported without modifications.

Here are some projects that harness the capabilities of `express-nlup-enhanced`:

* [Detector Morse](http://github.com/ExpressLens/detectormorse): This project is geared towards sentence boundary detection
* [Perceptronix Point Never](http://github.com/ExpressLens/PerceptronixPointNever): This project focusses on simple part of speech tagging
* [Where's Yr Head At](http://github.com/ExpressLens/WheresYrHeadAt): It involves simple transition-based dependency parsing