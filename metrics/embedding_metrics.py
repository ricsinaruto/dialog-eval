import numpy as np

from scipy.spatial import distance



# A helper class for embedding similarity metrics.
class EmbeddingMetrics():
  def __init__(self, vocab, distro, emb_dim):
    self.vocab = vocab
    self.emb_dim = emb_dim
    self.distro = distro

    self.metrics = {"embedding average": [],
                    "embedding extrema": [],
                    "embedding greedy": [],
                    "coherence": []}

  # Calculate embedding metrics.
  def update_metrics(self, source_words, resp_words, gt_words):
    avg_source = self.avg_embedding(source_words)
    avg_resp = self.avg_embedding(resp_words)
    avg_gt = self.avg_embedding(gt_words)

    # Check for zero vectors and compute cosine similarity.
    if np.count_nonzero(avg_resp):
      if np.count_nonzero(avg_source):
        self.metrics["coherence"].append(
          1 - distance.cosine(avg_source, avg_resp))
      if np.count_nonzero(avg_gt):
        self.metrics["embedding average"].append(
          1 - distance.cosine(avg_gt, avg_resp))

    # Compute extrema embedding metric.
    extrema_resp = self.extrema_embedding(resp_words)
    extrema_gt = self.extrema_embedding(gt_words)
    if np.count_nonzero(extrema_resp) and np.count_nonzero(extrema_gt):
      self.metrics["embedding extrema"].append(
        1 - distance.cosine(extrema_resp, extrema_gt))

    # Compute greedy embedding metric.
    one_side = self.greedy_embedding(gt_words, resp_words)
    other_side = self.greedy_embedding(resp_words, gt_words)

    if one_side and other_side:
      self.metrics["embedding greedy"].append((one_side + other_side) / 2)

  # Calculate the average word embedding of a sentence.
  def avg_embedding(self, words):
    vectors = []
    for word in words:
      vector = self.vocab.get(word)
      prob = self.distro.get(word)
      if vector:
        if prob:
          vectors.append(vector[0] * 0.001 / (0.001 + prob))
        else:
          vectors.append(vector[0] * 0.001 / (0.001 + 0))

    if vectors:
      return np.sum(np.array(vectors), axis=0) / len(vectors)
    else:
      return np.zeros(self.emb_dim)

  # Calculate the extrema embedding of a sentence.
  def extrema_embedding(self, words):
    vector = np.zeros(self.emb_dim)
    for word in words:
      vec = self.vocab.get(word)
      if vec:
        for i in range(self.emb_dim):
          if abs(vec[0][i]) > abs(vector[i]):
            vector[i] = vec[0][i]
    return vector

  # Calculate the greedy embedding from one side.
  def greedy_embedding(self, words1, words2):
    y_vec = np.zeros((self.emb_dim, 1))
    x_count = 0
    y_count = 0
    cos_sim = 0
    for word in words2:
      vec = self.vocab.get(word)
      if vec:
        norm = np.linalg.norm(vec[0])
        vector = vec[0] / norm if norm else vec[0]
        y_vec = np.hstack((y_vec, (vector.reshape((self.emb_dim, 1)))))
        y_count += 1

    for word in words1:
      vec = self.vocab.get(word)
      if vec:
        norm = np.linalg.norm(vec[0])
        if norm:
          cos_sim += np.max((vec[0] / norm).reshape((1, self.emb_dim)).dot(y_vec))
          x_count += 1

    if x_count > 0 and y_count > 0:
      return cos_sim / x_count
