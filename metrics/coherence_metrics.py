import numpy as np
from scipy.spatial import distance

from metrics.embedding_metrics import EmbeddingMetrics


class CoherenceMetrics(EmbeddingMetrics):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.metrics = {'coherence': []}

  # Calculate embedding metrics.
  def update_metrics(self, resp_words, gt_words, source_words):
    avg_source = self.avg_embedding(source_words)
    avg_resp = self.avg_embedding(resp_words)

    # Check for zero vectors and compute cosine similarity.
    if np.count_nonzero(avg_resp) and np.count_nonzero(avg_source):
        self.metrics['coherence'].append(
          1 - distance.cosine(avg_source, avg_resp))
