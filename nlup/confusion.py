# Copyright (c) 2013-2016 Kyle Gorman
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
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""Confusion matrix classes."""


from math import sqrt
from functools import partial
from collections import defaultdict


NAN = float("nan")
INF = float("inf")


class ConfusionMixin(object):
  """Confusion methods shared by all three non-abstract classes."""

  def batch_update(self, truths, guesses):
    for (truth, guess) in zip(truths, guesses):
      self.update(truth, guess)

  @property
  def confint(self):
    """Computes 95% binomial confidence intervals.

    This method computes the 95% binomial confidence interval for sample
    accuracy using the Wilson method. The lower bound is particularly
    useful in system ranking.

    The return value is a (lower_bound, upper_bound) tuple.
    """
    if not len(self):
      return (0., 1.)
    n = len(self)
    phat = self.accuracy
    z = 1.9599639845400538273879   # -qnorm(.05 / 2)
    zsq = z * z
    a1 = 1. / (1. + zsq / n)
    a2 = phat + zsq / (2 * n)
    a3 = z * sqrt(phat * (1. - phat) / n + zsq / (4 * n * n))
    return (a1 * (a2 - a3), a1 * (a2 + a3))


class Accuracy(ConfusionMixin):
  """Accuracy for classification tasks."""

  def __init__(self, correct=0, incorrect=0):
    self.correct = correct
    self.incorrect = incorrect

  def __repr__(self):
    return "<{} at 0x{:x}>".format(self.__class__.__name__, id(Self))

  def __str__(self):
    return "{:.4f}".format(self.accuracy)

  def __len__(self):
    return self.correct + self.incorrect

  def outcome(self, is_hit):
    if is_hit:
      self.correct += 1
    else:
      self.incorrect += 1

  def update(self, truth, guess):
    self.outcome(truth == guess)

  def __add__(self, right):
    """Combines two Accuracy objects."""
    return Accuracy(self.correct + right.correct,
                    self.incorrect + right.incorrect)

  @property
  def accuracy(self):
    return self.correct / len(self)


class BinaryConfusion(ConfusionMixin):
  """Binary confusion matrix for classification tasks."""

  def __init__(self, hit=True, tp=0, fp=0, fn=0, tn=0):
    self.hit = hit
    self.tp = tp
    self.fp = fp
    self.fn = fn
    self.tn = tn

  def __repr__(self):
    return "<{} at 0x{:x}>".format(self.__class__.__name__, id(Self))

  def pprint(self):
    """
    >>> cm = BinaryConfusion()
    >>> cm.tp = 5809125
    >>> cm.tn = 2235458
    >>> cm.fp = cm.fn = 1
    >>> cm.pprint()
    Truth | Guess
    ---------------------------------------
          |       Hit  