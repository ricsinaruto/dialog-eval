import math

from utils import utils


# https://arxiv.org/abs/1905.05471
class DivergenceMetrics():
  def __init__(self, vocab, gt_path):
    '''
    Params:
      :vocab: Vocabulary dictionary.
      :gt_path: Path to ground truth file.
    '''
    self.vocab = vocab
    self.gt_path = gt_path

    self.metrics = {'unigram-kl-div': [],
                    'bigram-kl-div': []}

  # Calculate kl divergence between between two distributions for a sentence.
  def update_metrics(self, resp, gt_words, source):
    '''
    Params:
      :resp_words: Response word list.
      :gt_words: Ground truth word list.
      :source_words: Source word list.
    '''
    uni_div = []
    bi_div = []
    word_count = len(gt_words)

    for i, word in enumerate(gt_words):
      if self.uni_distros['model'].get(word):
        word = word if self.vocab.get(word) else '<unk>'
        uni_div.append(math.log(self.uni_distros['gt'][word] /
                                self.uni_distros['model'][word], 2))

      if i < word_count - 1:
        word2 = gt_words[i + 1] if self.vocab.get(gt_words[i + 1]) else '<unk>'
        bigram = (word, word2)
        if self.bi_distros['model'].get(bigram):
          bi_div.append(math.log(self.bi_distros['gt'][bigram] /
                                 self.bi_distros['model'][bigram], 2))

    # Exclude divide by zero errors.
    if uni_div:
      self.metrics['unigram-kl-div'].append(sum(uni_div) / len(uni_div))
    if bi_div:
      self.metrics['bigram-kl-div'].append(sum(bi_div) / len(bi_div))

  # Get the distributions for test and ground truth data.
  def setup(self, filename):
    '''
    Params:
      :filename: Path to test responses.
    '''
    self.test_distro = {'uni': {}, 'bi': {}}
    self.gt_distro = {'uni': {}, 'bi': {}}
    utils.build_distro(self.vocab, self.test_distro, filename)
    utils.build_distro(self.vocab, self.gt_distro, self.gt_path)

    # Only keep intersection of test and ground truth distros.
    test, true = self.filter_distros(self.test_distro['uni'],
                                     self.gt_distro['uni'])
    self.uni_distros = {'model': test, 'gt': true}
    test, true = self.filter_distros(self.test_distro['bi'],
                                     self.gt_distro['bi'])
    self.bi_distros = {'model': test, 'gt': true}

  # Filter test and ground truth distributions, only keep intersection.
  def filter_distros(self, test, true):
    '''
    Params:
      :test: Test distribution.
      :true: Ground truth distribution.
    '''
    intersection = set.intersection(set(test.keys()), set(true.keys()))

    def probability_distro(distro):
      distro = dict(distro)
      for key in list(distro.keys()):
        if key not in intersection:
          del distro[key]
      return utils.convert_to_probs(distro)

    test = probability_distro(test)
    true = probability_distro(true)
    return test, true
