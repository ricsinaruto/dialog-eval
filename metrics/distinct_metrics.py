
# A helper class for distinct metrics.
class DistinctMetrics():
  def __init__(self, test_distro, gt_distro):
    self.test_distro = test_distro
    self.gt_distro = gt_distro
    self.metrics = {"distinct-1": [],
                    "distinct-2": [],
                    "distinct-1 ratio": [],
                    "distinct-2 ratio": []}

  def distinct(self, distro):
    return len(distro) / sum(list(distro.values()))

  def calculate_metrics(self):
    self.metrics["distinct-1"].append(self.distinct(self.test_distro["uni"]))
    self.metrics["distinct-2"].append(self.distinct(self.test_distro["bi"]))
    self.metrics["distinct-1 ratio"].append(
      self.metrics["distinct-1"][-1] / self.distinct(self.gt_distro["uni"]))
    self.metrics["distinct-2 ratio"].append(
      self.metrics["distinct-2"][-1] / self.distinct(self.gt_distro["bi"]))