from .span import Span


class Annotation(object):
  def __init__(self):
    self.type = ''
    self.span = Span()
    # TODO: Featureset implementations