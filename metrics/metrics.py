import math
import sys
import os
import numpy as np


class Metrics:
  def __init__(self, config):
    """
    Params:
      :test_responses_path: Path to the model responses on test set.
    """
    # Save all filenames of test responses
    self.config = config
    filenames = []
    if os.path.isdir(config.test_responses):
      output_path = os.path.join(config.test_responses, 'metrics.txt')
      for filename in os.listdir(config.test_responses):
        filenames.append(filename)
    else:
      filenames.append(config.test_responses.split('/')[-1])
      output_path = os.path.join('/'.join(config.test_responses.split('/')[:-1]), 'metrics.txt')

    self.which_metrics = dict([(key, 0) for key in config.metrics])
    self.metrics = dict([(name, dict([(key, [0, 0, 0]) for key in config.metrics])) for name in filenames])

    self.project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    self.output_path = os.path.join(self.project_path, output_path)
    
    # Unigram and bigram probabilities based on train, model and test data.
    self.vocab = {}
    self.train_distro = {"uni": {}, "bi": {}}
    self.test_distro = {"uni": {}, "bi": {}}
    self.gt_distro = {"uni": {}, "bi": {}}
    

    # Build the distributions.
    # should be separate for test responses
    self.build_distributions()

    

    # Initialize metrics.
    self.response_len = {"length": []}
    self.entropies = EntropyMetrics(
      self.vocab, self.train_distro, self.filtered_uni, self.filtered_bi)
    self.embedding = EmbeddingMetrics(
      self.vocab, self.train_distro["uni"], self.emb_dim)
    self.distinct = DistinctMetrics(self.test_distro, self.gt_distro)
    self.bleu = BleuMetrics()

  # Count words, load vocab files and build distributions.
  def build_distributions(self):
    # Build the word vectors.
    with open(self.paths["vector_vocab"]) as file:
      for line in file:
        line_as_list = line.split()
        vector = np.array([float(num) for num in line_as_list[1:]])
        self.vocab[line_as_list[0]] = [vector]

    self.emb_dim = list(self.vocab.values())[0][0].size

    # Extend the remaining vocab.
    with open(self.paths["text_vocab"]) as file:
      for line in file:
        line = line.strip()
        if not self.vocab.get(line):
          self.vocab[line] = [np.zeros(self.emb_dim)]

    # Go through the train file and build word and bigram frequencies.
    def build_distro(distro, path):
      with open(path) as file:
        for line in file:
          words = line.split()
          word_count = len(words)
          for i, word in enumerate(words):
            word = word if self.vocab.get(word) else "<unk>"
            w_in_dict = distro["uni"].get(word)
            distro["uni"][word] = distro["uni"][word] + 1 if w_in_dict else 1

            # Bigrams.
            if i < word_count - 1:
              word2 = words[i + 1] if self.vocab.get(words[i + 1]) else "<unk>"
              bi = (word, word2)
              bigram_in_dict = distro["bi"].get(bi)
              distro["bi"][bi] = distro["bi"][bi] + 1 if bigram_in_dict else 1

    # Converts frequency dict to probabilities
    def convert_to_probs(freq_dict):
      num_words = sum(list(freq_dict.values()))
      return dict([(key, val / num_words) for key, val in freq_dict.items()])

    # Filter test and ground truth distributions, only keep intersection.
    def filter_distros(test, true):
      intersection = set.intersection(set(test.keys()), set(true.keys()))

      def probability_distro(distro):
        distro = dict(distro)
        for key in list(distro.keys()):
          if key not in intersection:
            del distro[key]
        return convert_to_probs(distro)

      test = probability_distro(test)
      true = probability_distro(true)
      return test, true

    # Build the three distributions.
    build_distro(self.train_distro, self.paths["train_source"])
    build_distro(self.test_distro, self.paths["test_responses"])
    build_distro(self.gt_distro, self.paths["gt_responses"])

    # Get probabilities for train distro.
    self.train_distro["uni"] = convert_to_probs(self.train_distro["uni"])
    self.train_distro["bi"] = convert_to_probs(self.train_distro["bi"])

    # Only keep intersection of test and ground truth distros.
    test, true = filter_distros(self.test_distro["uni"], self.gt_distro["uni"])
    self.filtered_uni = {"model": test, "gt": true}
    test, true = filter_distros(self.test_distro["bi"], self.gt_distro["bi"])
    self.filtered_bi = {"model": test, "gt": true}

  # Compute all metrics.
  def run(self):
    sources = open(self.paths["test_source"])
    responses = open(self.paths["test_responses"])
    gt_responses = open(self.paths["gt_responses"])

    # Loop through the test and ground truth responses, and calculate metrics.
    for source, response, target in zip(sources, responses, gt_responses):
      gt_words = target.split()
      resp_words = response.split()
      source_words = source.split()
      self.response_len["length"].append(len(resp_words))

      # Calculate metrics.
      self.entropies.update_metrics(resp_words, gt_words)
      self.embedding.update_metrics(source_words, resp_words, gt_words)
      self.bleu.update_metrics(resp_words, gt_words)
    self.distinct.calculate_metrics()

    sources.close()
    gt_responses.close()
    responses.close()

  # Compute mean, std and confidence, and write the given metric to file.
  def write_metrics(self):
    metrics = {**self.response_len,
               **self.entropies.metrics,
               **self.embedding.metrics,
               **self.distinct.metrics,
               **self.bleu.metrics}

    with open(self.paths["output"], "w") as output:
      for name, metric in metrics.items():
        length = len(metric)
        avg = sum(metric) / length
        std = np.std(metric) if length > 1 else 0

        # 95% confidence interval (t=1.97)
        confidence = 1.97 * std / math.sqrt(length)

        # Write the metric to file.
        m = name + ": " + str(avg) + " " + str(std) + " " + str(confidence)
        print(m)
        output.write(m + '\n')
