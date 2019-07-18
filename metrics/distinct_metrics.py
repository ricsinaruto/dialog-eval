from utils import utils


# https://www.aclweb.org/anthology/N16-1014
class DistinctMetrics():
  def __init__(self, vocab):
    '''
    Params:
      :vocab: Vocabulary dictionary.
    '''
    self.vocab = vocab
    self.metrics = {'distinct-1': [],
                    'distinct-2': []}

  # Calculate the distinct value for a distribution.
  def distinct(self, distro):
    return len(distro) / sum(list(distro.values()))

  # Calculate distinct metrics for a given file.
  def calculate_metrics(self, filename):
    test_distro = {'uni': {}, 'bi': {}}
    utils.build_distro(self.vocab, test_distro, filename)

    self.metrics['distinct-1'].append(self.distinct(test_distro['uni']))
    self.metrics['distinct-2'].append(self.distinct(test_distro['bi']))

  # Ghost function.
  def update_metrics(self, a, s, d):
    pass
