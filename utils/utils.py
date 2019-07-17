# Converts frequency dict to probabilities
def convert_to_probs(freq_dict):
  num_words = sum(list(freq_dict.values()))
  return dict([(key, val / num_words) for key, val in freq_dict.items()])


# Go through the train file and build word and bigram frequencies.
def build_distro(vocab, distro, path, probs=False):
  with open(path) as file:
    for line in file:
      words = line.split()
      word_count = len(words)
      for i, word in enumerate(words):
        word = word if vocab.get(word) else '<unk>'
        w_in_dict = distro['uni'].get(word)
        distro['uni'][word] = distro['uni'][word] + 1 if w_in_dict else 1

        # Bigrams.
        if i < word_count - 1:
          word2 = words[i + 1] if vocab.get(words[i + 1]) else '<unk>'
          bi = (word, word2)
          bigram_in_dict = distro['bi'].get(bi)
          distro['bi'][bi] = distro['bi'][bi] + 1 if bigram_in_dict else 1

  if probs:
    distro['uni'] = convert_to_probs(distro['uni'])
    distro['bi'] = convert_to_probs(distro['bi'])
