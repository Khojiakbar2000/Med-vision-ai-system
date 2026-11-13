Medical Segmentation Decathlon Task01 BrainTumour label values are 0 background, 1 edema, 2 non-enhancing tumor, and 3 enhancing tumor; its four image channels are ordered FLAIR, T1w, T1gd, T2w. For Task01, the evaluation regions are WT={1,2,3}, TC={2,3}, and ET={3}; the application model uses WT={1,2,4}, TC={1,4}, and ET={4}.

In evaluation against this application's MONAI bundle, Task01 enhancing label 3 is compared with model label 4 only when the documented msd_task01 reference convention is explicitly selected. The reference array itself is not rewritten.

The project has verified a single Task01 training-split case. Recorded per-case Dice and IoU values are not estimates of population performance, and independence from the MONAI bundle's BraTS 2018 training cohort is unknown. Do not generalize that single-case result to other scanners, protocols, populations, or clinical use.
