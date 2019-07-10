import sys
import argparse


from metrics.metrics import Metrics
from config import Config


def main():
  config = Config()
  parser = argparse.ArgumentParser(
    description='Code for filtering methods in: arxiv.org/abs/1905.05471. ' +
                'These arguments can also be set in config.py, ' +
                'and will be saved to the output directory.')
  parser.add_argument('-d', '--data_dir', default=config.data_dir,
                      help='Directory containing the dataset in these files:' +
                      ' (trainSource.txt, trainTarget.txt, devSource.txt, ' +
                      'devTarget.txt, testSource.txt, testTarget.txt, ' +
                      'vocab.txt)',
                      metavar='')
  parser.add_argument('-o', '--output_dir', default=config.output_dir,
                      help='Save here the filtered data and any output',
                      metavar='')
  parser.add_argument('-l', '--load_config', default=config.load_config,
                      help='Path to load config from file, or leave empty ' +
                      '(default: %(default)s)',
                      metavar='')
  parser.add_argument('-fs', '--filter_split', default=config.filter_split,
                      help='Data split to filter, \'full\' filters ' +
                      'all splits (choices: %(choices)s)',
                      metavar='', choices=['full', 'train', 'dev', 'test'])
  parser.add_argument('-ct', '--cluster_type', default=config.cluster_type,
                      help='Clustering method (choices: %(choices)s)',
                      metavar='',
                      choices=['identity', 'avg_embedding', "sent2vec"])
  parser.add_argument('-sc', '--source_clusters',
                      default=config.source_clusters,
                      help='Number of source clusters in case of Kmeans',
                      metavar='', type=int)
  parser.add_argument('-tc', '--target_clusters',
                      default=config.target_clusters,
                      help='Number of target clusters in case of Kmeans',
                      metavar='', type=int)
  parser.add_argument('-u', '--unique', default=config.unique,
                      help='Whether to cluster only unique sentences ' +
                      '(default: %(default)s)',
                      metavar='', type=bool)

  parser.parse_args(namespace=config)

  m = Metrics(config)
  m.run()


if __name__ == "__main__":
  main()
