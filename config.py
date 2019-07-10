# These can also be set as arguments via the command line.
class Config:
  train_source = 'data/DailyDialog/baseline/trainSource.txt'
  test_source = 'data/DailyDialog/baseline/testSource.txt'
  test_target = 'data/DailyDialog/baseline/testTarget.txt'
  text_vocab = 'data/DailyDialog/baseline/vocab.txt'
  vector_vocab = 'data/DailyDialog/baseline/vocab.npy'
  test_responses = 'data/DailyDialog/baseline/testTarget.txt'
  metrics = {
    'length': 1,
    'per-unigram-entropy': 1,
    'per-bigram-entropy': 1,
    'utterance-unigram-entropy': 1,
    'utterance-bigram-entropy': 1,
    'unigram-kl-div': 1,
    'bigram-kl-div': 1,
    'embedding-average': 1,
    'embedding-extrema': 1,
    'embedding-greedy': 1,
    'coherence': 1,
    'distinct-1': 1,
    'distinct-2': 1,
    'bleu-1': 1,
    'bleu-2': 1,
    'bleu-3': 1,
    'bleu-4': 1
  }