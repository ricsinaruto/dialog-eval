from utils import utils


# A helper class for distinct metrics.
class DistinctMetrics():
  def __init__(self, vocab):
    self.vocab = vocab
    self.metrics = {'distinct-1': [],
                    'distinct-2': []}

  def distinct(self, distro):
    return len(distro) / sum(list(distro.values()))

  def calculate_metrics(self, filename):
    test_distro = {'uni': {}, 'bi': {}}
    utils.build_distro(self.vocab, test_distro, filename)

    self.metrics['distinct-1'].append(self.distinct(test_distro['uni']))
    self.metrics['distinct-2'].append(self.distinct(test_distro['bi']))

  def update_metrics(self, a, s, d):
    pass
