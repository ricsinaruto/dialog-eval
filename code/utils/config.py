folder = 'data/DailyDialogCurated/'


# These can also be set as arguments via the command line.
class Config:
  bleu_smoothing = 4  # Smoothing method for bleu calculation.
  t = 1.97  # t value for confidence level calculation
  train_source = folder + 'trainSource.txt'
  test_source = folder + 'testSource.txt'
  test_target = folder + 'testTarget.txt'
  text_vocab = folder + 'dataset/vocab.txt'
  vector_vocab = folder + 'dataset/vocab.npy'
  test_responses = folder + 'gpt2/opensubtitles_transfer/valmin_finetuned_processed.txt'
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
