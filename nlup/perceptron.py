# Copyright (C) 2014-2016 Kyle Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# PLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""Perceptron-like classifers, including:

* `BinaryPerceptron`: binary perceptron classifier
* `Perceptron`: multiclass perceptron classifier
* `SequencePerceptron`: multiclass perceptron for sequence tagging
* `BinaryAveragedPerceptron`: binary averaged perceptron classifier
* `AveragedPerceptron`: multiclass averaged perceptron
* `SequenceAveragedPerceptron`: multiclass averaged perceptron for
   sequence tagging
"""


import logging

from collections import defaultdict
from collections import namedtuple
from functools import partial
from operator import itemgetter
from random import Random

from .confusion import Accuracy
from .decorators import reversify
from .decorators import tupleify
from .jsonable import JSONable
from .timer import Timer


INF = float("inf")
ORDER = 0
EPOCHS = 1


class Classifier(JSONable):
  """Mixin for shared classifier methods."""

  def fit(self, Y, Phi, epochs=EPOCHS, alpha=1):
    data = list(zip(Y, Phi))  # Which is a copy.
    logging.info("Starting {} epoch(s) of training.".format(epochs))
    for epoch in range(1, 1 + epochs):
      logging.info("Epoch {:>2}.".format(epoch))
      accuracy = Accuracy()
      self.random.shuffle(data)
      with Timer():
        for (y, phi) in data:
          yhat = self.fit_one(y, phi, alpha)
          accuracy.update(y, yhat)
      logging.info("Accuracy: {!s}".format(accuracy))
    self.finalize()

  def finalize(self):
    pass


class BinaryPerceptron(Classifier):
  """Binary perceptron classifier."""

  def __init__(self, seed=None):
    self.random = Random(seed)
    self.weights = defaultdict(int)

  def score(self, phi):
    """Gets score for a feature vector."""
    return sum(self.weights[feature] for feature in phi)

  def predict(self, phi):
    """Predicts binary decision for a feature vector."""
    return self.score(phi) >= 0

  def fit_one(self, y, phi, alpha=1):
    yhat = self.predict(phi)
    if y != yhat:
      self.update(y, phi, alpha)
    return yhat

  def update(self, y, phi, alpha=1):
    """Rewards correct observation for a feature vector."""
    assert y in (True, False)
    assert 0. < alpha <= 1.
    if y is False:
      alpha *= -1
    for phi_i in phi:
      self.weights[phi_i] += alpha


class Perceptron(Classifier):
  """Multiclass perceptron with sparse binary feature vectors.

  Each class (i.e., label, outcome) is represented as a hashable item, such as a
  string. Features are represented as hashable objects (preferably strings, as
  Python dictionaries have been aggressively optimized for this case). Presence
  of a feature indicates that that feature is "firing" and absence indicates
  that it is not firing. This class is primarily to be used as an abstract base
  class; in most cases, the regularization and stability afforded by the
  averaged perceptron (`AveragedPerceptron`) will be worth it.

  The perceptron was first proposed in the following paper:

  F. Rosenblatt. 1958. The perceptron: A probabilistic model for information
  storage and organization in the brain. Psychological Review 65(6): 386-408.
  """

  def __init__(self, classes=(), seed=None):
    s