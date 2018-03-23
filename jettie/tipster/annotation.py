from .span import Span


class Annotation(object):
  def __init__(self, type, span):
    self.type = type
    self.span = span

  def __repr__(self):
    return '%s@%s' % (self.type, self.span)

  # TODO: implement method which returns if an annotation is non-key, e.g. `<other>`