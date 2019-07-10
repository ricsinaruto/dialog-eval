# A helper class for entropy-based metrics.
class EntropyMetrics():
  def __init__(self, vocab, train_distro, uni_distros, bi_distros):
    self.vocab = vocab
    self.train_distro = train_distro
    self.uni_distros = uni_distros
    self.bi_distros = bi_distros

    self.metrics = {"word unigram entropy": [],
                    "word bigram entropy": [],
                    "utterance unigram entropy": [],
                    "utterance bigram entropy": [],
                    "unigram kl divergence": [],
                    "bigram kl divergence": []}

  def update_metrics(self, resp_words, gt_words):
    uni_entropy = []
    bi_entropy = []
    word_count = len(resp_words)
    for i, word in enumerate(resp_words):
      # Calculate unigram entropy.
      word = word if self.vocab.get(word) else "<unk>"
      probability = self.train_distro["uni"].get(word)
      if probability:
        uni_entropy.append(math.log(probability, 2))

      # Calculate bigram entropy.
      if i < word_count - 1:
        w = resp_words[i + 1] if self.vocab.get(resp_words[i + 1]) else "<unk>"
        probability = self.train_distro["bi"].get((word, w))
        if probability:
          bi_entropy.append(math.log(probability, 2))

    # Check if lists are empty.
    if uni_entropy:
      entropy = -sum(uni_entropy)
      self.metrics["word unigram entropy"].append(entropy / len(uni_entropy))
      self.metrics["utterance unigram entropy"].append(entropy)
    if bi_entropy:
      entropy = -sum(bi_entropy)
      self.metrics["word bigram entropy"].append(entropy / len(bi_entropy))
      self.metrics["utterance bigram entropy"].append(entropy)

    # KL-divergence
    self.calc_kl_divergence(gt_words)

  # Calculate kl divergence between between two distributions for a sentence.
  def calc_kl_divergence(self, gt_words):
    uni_div = []
    bi_div = []
    word_count = len(gt_words)

    for i, word in enumerate(gt_words):
      if self.uni_distros["model"].get(word):
        word = word if self.vocab.get(word) else "<unk>"
        uni_div.append(math.log(self.uni_distros["gt"][word] /
                                self.uni_distros["model"][word], 2))

      if i < word_count - 1:
        word2 = gt_words[i + 1] if self.vocab.get(gt_words[i + 1]) else "<unk>"
        bigram = (word, word2)
        if self.bi_distros["model"].get(bigram):
          bi_div.append(math.log(self.bi_distros["gt"][bigram] /
                                 self.bi_distros["model"][bigram], 2))

    # Exclude divide by zero errors.
    if uni_div:
      self.metrics["unigram kl divergence"].append(sum(uni_div) / len(uni_div))
    if bi_div:
      self.metrics["bigram kl divergence"].append(sum(bi_div) / len(bi_div))