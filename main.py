import sys
import argparse


from metrics.metrics import Metrics
from config import Config


def main():
  config = Config()
  parser = argparse.ArgumentParser(
    description='Code for evaluating dialog models\' responses with' +
                '17 evaluation metrics (arxiv.org/abs/1905.05471)')
  parser.add_argument('-tns', '--train_source', default=config.train_source,
                      help='Path to the train source file, where each line ' +
                      'corresponds to one train input',
                      metavar='')
  parser.add_argument('-tts', '--test_source', default=config.test_source,
                      help='Path to the test source file, where each line ' +
                      'corresponds to one test input',
                      metavar='')
  parser.add_argument('-ttt', '--test_target', default=config.test_target,
                      help='Path to the test target file, where each line ' +
                      'corresponds to one test target',
                      metavar='')
  parser.add_argument('-r', '--test_responses', default=config.test_responses,
                      help='Path to the test model responses file',
                      metavar='')
  parser.add_argument('-tv', '--text_vocab', default=config.text_vocab,
                      help='A file where each line is a word in the vocab',
                      metavar='')
  parser.add_argument('-vv', '--vector_vocab', default=config.vector_vocab,
                      help='A file where each line is a word in the vocab ' +
                      'followed by a vector',
                      metavar='')
  parser.add_argument('-s', '--bleu_smoothing', default=config.bleu_smoothing,
                      help='Bleu smoothing method (choices: %(choices)s)',
                      metavar='',
                      choices=[0, 1, 2, 3, 4, 5, 6, 7])
  parser.add_argument('-t', '--t', default=config.t,
                      help='t value for confidence level calculation ' +
                      '(default: %(default)s)',
                      metavar='', type=int)

  parser.parse_args(namespace=config)

  m = Metrics(config)
  m.run()


if __name__ == "__main__":
  main()
