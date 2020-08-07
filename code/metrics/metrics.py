import math
import sys
import os
import numpy as np
import requests
import zipfile
from collections import Counter
from clint.textui import progress

from metrics.bleu_metrics import BleuMetrics
from metrics.distinct_metrics import DistinctMetrics
from metrics.entropy_metrics import EntropyMetrics
from metrics.embedding_metrics import EmbeddingMetrics
from metrics.divergence_metrics import DivergenceMetrics
from metrics.coherence_metrics import CoherenceMetrics
from utils import utils


class Metrics:
  def __init__(self, config):
    '''
    Params:
      :config: A Config instance containing arguments.
    '''
    self.project_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '..', '..')
    self.test_responses = os.path.join(self.project_path,
                                       config.test_responses)

    if not os.path.exists(self.test_responses):
      print('Can\' find test responses at ' + self.test_responses +
        ', please specify the path.')
      sys.exit()

    self.config = config
    self.distro = {'uni': {}, 'bi': {}}
    self.vocab = {}

    # Save all filenames of test responses and build output path.
    filenames = []
    if os.path.isdir(self.test_responses):
      self.input_dir = self.test_responses
      self.output_path = os.path.join(self.test_responses, 'metrics.txt')
      for filename in os.listdir(self.test_responses):
        filenames.append(os.path.join(self.test_responses, filename))
    else:
      self.input_dir = '/'.join(self.test_responses.split('/')[:-1])
      filenames.append(self.test_responses)
      self.output_path = os.path.join(self.input_dir, 'metrics.txt')

    # Initialize metrics and a bool dict for which metrics should be selected.
    self.which_metrics = dict(config.metrics)
    self.metrics = dict([(name, dict(
      [(key, []) for key in config.metrics])) for name in filenames])

    # Absolute path.
    self.train_source = os.path.join(self.project_path, config.train_source)
    self.test_source = os.path.join(self.project_path, config.test_source)
    self.test_target = os.path.join(self.project_path, config.test_target)
    self.text_vocab = os.path.join(self.project_path, config.text_vocab)
    self.vector_vocab = os.path.join(self.project_path, config.vector_vocab)

    # Check which metrics we can compute.
    if not os.path.exists(self.train_source):
      print('Can\'t find train data at ' + self.train_source + ', entropy ' +
        'metrics, \'coherence\' and \'embedding-average\' won\'t be computed.')
      self.delete_from_metrics(['entropy', 'average', 'coherence'])
    if not os.path.exists(self.test_source):
      print('Can\' find test sources at ' + self.test_source +
        ', \'coherence\' won\'t be computed.')
      self.delete_from_metrics(['coherence'])
    if not os.path.exists(self.test_target):
      print('Can\' find test targets at ' + self.test_target +
        ', embedding, kl divergence, and bleu metrics won\'t be computed.')
      self.delete_from_metrics(['kl-div', 'embedding', 'bleu'])
    if not os.path.exists(self.vector_vocab):
      print('File containing word vectors not found in ' + self.vector_vocab)
      print('If you would like to use FastText embeddings press \'y\'')
      if input() == 'y':
        self.get_fast_text_embeddings()
      else:
        print('Embedding metrics and \'coherence\' won\'t be computed.')
        self.delete_from_metrics(['coherence', 'embedding'])
    if not os.path.exists(self.text_vocab):
      print('No vocab file named \'vocab.txt\' found in ' + self.text_vocab)
      if os.path.exists(self.train_source):
        print('Building vocab from data.')
        self.text_vocab = os.path.join(self.input_dir, 'vocab.txt')
        self.get_vocab()

    # Build vocab and train data distribution if needed.
    if os.path.exists(self.text_vocab):
      self.build_vocab()
    if os.path.exists(self.train_source):
      utils.build_distro(self.distro, self.train_source, self.vocab, True)

    self.objects = {}
    self.objects['distinct'] = DistinctMetrics(self.vocab)
    # Initialize metric objects.
    if self.these_metrics('entropy'):
      self.objects['entropy'] = EntropyMetrics(self.vocab, self.distro)
    if self.these_metrics('kl-div'):
      self.objects['divergence'] = DivergenceMetrics(self.vocab,
                                                     self.test_target)
    if self.these_metrics('embedding'):
      self.objects['embedding'] = EmbeddingMetrics(
        self.vocab,
        self.distro['uni'],
        self.emb_dim,
        self.which_metrics['embedding-average'])
    if self.these_metrics('coherence'):
      self.objects['coherence'] = CoherenceMetrics(
        self.vocab, self.distro['uni'], self.emb_dim)
    if self.these_metrics('bleu'):
      self.objects['bleu'] = BleuMetrics(config.bleu_smoothing)

  # Whether these metrics are activated.
  def these_metrics(self, metric):
    activated = False
    for key in self.which_metrics:
      if metric in key and self.which_metrics[key]:
        activated = True

    return activated

  # Download data from fasttext.
  def download_fasttext(self):
    # Open the url and download the data with progress bars.
    data_stream = requests.get('https://dl.fbaipublicfiles.com/fasttext/' +
      'vectors-english/wiki-news-300d-1M.vec.zip', stream=True)
    zipped_path = os.path.join(self.input_dir, 'fasttext.zip')

    with open(zipped_path, 'wb') as file:
      total_length = int(data_stream.headers.get('content-length'))
      for chunk in progress.bar(data_stream.iter_content(chunk_size=1024),
                                expected_size=total_length / 1024 + 1):
        if chunk:
          file.write(chunk)
          file.flush()

    # Extract file.
    zip_file = zipfile.ZipFile(zipped_path, 'r')
    zip_file.extractall(self.input_dir)
    zip_file.close()

  # Generate a vocab from data files.
  def get_vocab(self):
    vocab = []
    if not os.path.exists(self.train_source):
      print('No train data, can\'t build vocab file.')
      sys.exit()

    with open(self.text_vocab, 'w', encoding='utf-8') as file:
      with open(self.train_source, encoding='utf-8') as in_file:
        for line in in_file:
          vocab.extend(line.split())
      file.write('\n'.join(list(Counter(vocab))))

  # Download FastText word embeddings.
  def get_fast_text_embeddings(self):
    if not os.path.exists(self.text_vocab):
      print('No vocab file named \'vocab.txt\' found in ' + self.text_vocab)
      print('Building vocab from data.')
      self.text_vocab = os.path.join(self.input_dir, 'vocab.txt')
      self.get_vocab()

    fasttext_path = os.path.join(self.input_dir, 'cc.' + self.config.lang + '.300.vec')
    if not os.path.exists(fasttext_path):
      self.download_fasttext()

    vocab = [line.strip('\n') for line in open(self.text_vocab, encoding='utf-8')]
    self.vector_vocab = os.path.join(self.input_dir, 'vocab.npy')

    # Save the vectors for words in the vocab.
    with open(fasttext_path, errors='ignore', encoding='utf-8') as in_file:
      with open(self.vector_vocab, 'w', encoding='utf-8') as out_file:
        vectors = {}
        for line in in_file:
          tokens = line.strip().split()
          if len(tokens) == 301:
            vectors[tokens[0]] = line
          elif tokens[1] == '»':
            vectors[tokens[0]] = tokens[0] + ' ' + ' '.join(tokens[2:]) + '\n'

        for word in vocab:
          try:
            out_file.write(vectors[word])
          except KeyError:
            pass

  # Set to 0 a given list of metrics in the which_metrics dict.
  def delete_from_metrics(self, metric_list):
    for key in self.which_metrics:
      for metric in metric_list:
        if metric in key:
          self.which_metrics[key] = 0

  # Build a vocabulary.
  def build_vocab(self):
    # Build the word vectors if possible.
    try:
      with open(self.vector_vocab, encoding='utf-8') as file:
        for line in file:
          tokens = line.split()
          self.vocab[tokens[0]] = [np.array(list(map(float, tokens[1:])))]

      self.emb_dim = list(self.vocab.values())[0][0].size
    except FileNotFoundError:
      self.emb_dim = 1

    # Extend the remaining vocab.
    with open(self.text_vocab, encoding='utf-8') as file:
      for line in file:
        line = line.strip()
        if not self.vocab.get(line):
          self.vocab[line] = [np.zeros(self.emb_dim)]

  # Compute all metrics for all files.
  def run(self):
    for filename in self.metrics:
      responses = open(filename, encoding='utf-8')
      # If we don't need these just open a dummy file.
      sources = open(self.test_source, encoding='utf-8') \
        if os.path.exists(self.test_source) else open(filename, encoding='utf-8')
      gt_responses = open(self.test_target, encoding='utf-8') \
        if os.path.exists(self.test_target) else open(filename, encoding='utf-8')

      # Some metrics require pre-computation.
      self.objects['distinct'].calculate_metrics(filename)
      if self.objects.get('divergence'):
        self.objects['divergence'].setup(filename)

      # Loop through the test and ground truth responses, calculate metrics.
      for source, response, target in zip(sources, responses, gt_responses):
        gt_words = target.split()
        resp_words = response.split()
        source_words = source.split()
        self.metrics[filename]['length'].append(len(resp_words))

        for key in self.objects:
          self.objects[key].update_metrics(resp_words, gt_words, source_words)

      sources.close()
      gt_responses.close()
      responses.close()

      # Save individual metrics to self.metrics
      for key in self.objects:
        for metric_name, metric in self.objects[key].metrics.items():
          self.metrics[filename][metric_name] = list(metric)
          self.objects[key].metrics[metric_name].clear()

    self.write_metrics()

  # Compute mean, std and confidence, and write all metrics to output file.
  def write_metrics(self):
    with open(self.output_path, 'w') as output:
      output.write('filename ')
      output.write(' '.join([k for k, v in self.which_metrics.items() if v]))
      output.write('\n')

      ''' The first row contains the names of the metrics, then each row
      contains the name of the file and its metrics separated by spaces.
      Each metric contains 3 numbers separated by ',': mean,std,confidence. '''
      for filename, metrics in self.metrics.items():
        output.write(filename.split('/')[-1] + ' ')
        for metric_name, metric in metrics.items():
          if self.which_metrics[metric_name]:
            length = len(metric)
            avg = sum(metric) / length
            std = np.std(metric) if length > 1 else 0
            confidence = self.config.t * std / math.sqrt(length)

            # Write the metric to file.
            m = str(avg) + ',' + str(std) + ',' + str(confidence)
            output.write(m + ' ')
        output.write('\n')
