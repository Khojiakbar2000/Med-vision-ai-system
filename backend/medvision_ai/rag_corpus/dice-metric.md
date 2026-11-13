MONAI provides DiceMetric for evaluating segmentation overlap. For a binary region, Dice is twice the intersection of prediction and reference divided by the sum of their voxel counts. Dice is 1 for identical non-empty masks and 0 for disjoint masks; empty-mask handling depends on the evaluation implementation and must be stated.

A Dice value is an overlap statistic for one specified prediction and reference. It is not a probability of correctness or a measure of clinical benefit. Report the region definition and dataset convention alongside the score.
