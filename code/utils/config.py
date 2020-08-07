folder = 'data/open_gut/en/'


# These can also be set as arguments via the command line.
class Config:
  bleu_smoothing = 4  # Smoothing method for bleu calculation.
  t = 1.97  # t value for confidence level calculation
  train_source = folder + 'trainSource.txt'
  test_source = folder + 'testSource.txt'
  test_target = folder + 'testTarget.txt'
  text_vocab = folder + 'vocab.txt'
  vector_vocab = folder + 'vocab.npy'
  test_responses = folder + 'responses'
  lang = 'en'
  metrics = {
    'length': 0,
    'per-unigram-entropy': 0,
    'per-bigram-entropy': 0,
    'utterance-unigram-entropy': 0,
    'utterance-bigram-entropy': 0,
    'unigram-kl-div': 0,
    'bigram-kl-div': 0,
    'embedding-average': 0,
    'embedding-extrema': 0,
    'embedding-greedy': 0,
    'coherence': 0,
    'distinct-1': 0,
    'distinct-2': 0,
    'bleu-1': 0,
    'bleu-2': 0,
    'bleu-3': 0,
    'bleu-4': 1
  }
