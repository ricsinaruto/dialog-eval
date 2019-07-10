from nltk.translate import bleu_score


class BleuMetrics():
  def __init__(self):
    self.metrics = {"bleu-1": [], "bleu-2": [], "bleu-3": [], "bleu-4": []}
    self.smoothing = bleu_score.SmoothingFunction().method4

  def update_metrics(self, resp, gt):
    try:
      self.metrics["bleu-1"].append(
        bleu_score.sentence_bleu([gt], resp, weights=(1, 0, 0, 0), smoothing_function=self.smoothing))
      self.metrics["bleu-2"].append(
        bleu_score.sentence_bleu([gt], resp, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing))
      self.metrics["bleu-3"].append(
        bleu_score.sentence_bleu([gt], resp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing))
      self.metrics["bleu-4"].append(
        bleu_score.sentence_bleu([gt], resp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing))
    except (KeyError, ZeroDivisionError):
      self.metrics["bleu-1"].append(0)
      self.metrics["bleu-2"].append(0)
      self.metrics["bleu-3"].append(0)
      self.metrics["bleu-4"].append(0)