# TODO: determine whether needed to add an Document pointer
from sys import stderr


class Span(object):
  """ The span of an annotation. """
  def __init__(self, s=0, e=0):
    try:
      if not (isinstance(s, int) and isinstance(e, int)):
        raise TypeError
    except TypeError:
      print('TypeError: both start and end should be specified of Integer type', file=stderr)
      exit(1)
    self.start = s
    self.end = e

  def __eq__(self, other):
    if self.start == other.start and self.end == other.end:
      return True
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    return '(%d, %d)' % (self.start, self.end)