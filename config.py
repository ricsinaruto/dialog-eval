# These can also be set as arguments via the command line.
class Config:
  bleu_smoothing = 4
  t = 1.97  # t value for confidence level calculation
  train_source = 'data/DailyDialog/baseline/trainSource.txt'
  test_source = 'data/DailyDialog/baseline/testSource.txt'
  test_target = 'data/DailyDialog/baseline/testTarget.txt'
  text_vocab = 'data/DailyDialog/vocab.txt'
  vector_vocab = 'data/DailyDialog/vocab.npy'
  test_responses = 'decode/temp'
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
