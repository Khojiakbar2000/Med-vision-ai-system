# Codex handoff: MOTUM overprediction investigation

## Current state

- The verified MOTUM research subset and experiment outputs remain outside the Git repository under `~/MedVision-Research/`.
- The three-epoch model is `experiments/MOTUM-3epoch/motum-three-epoch.pt`; the original MONAI checkpoint and prior step-one checkpoint were not modified.
- The four validation subjects were used for patch-only diagnostics. The five untouched evaluation subjects were not opened. No full-volume inference was run.
- Head and threshold diagnostics are in `experiments/MOTUM-overprediction-diagnosis/head-distributions.json`.
- A fresh seeded reconstruction does not reproduce the historical random head exactly; the initial RNG state was not saved. Overprediction is established by step one and persists through three epochs.
- Both channels predict nearly all foreground at threshold 0.5 on validation foreground and background patches. CE-T1c background mean probability is about 0.79 after three epochs; NCE-FLAIR background mean is about 0.73.
- Channel order, normalization, mask alignment, and loss-gradient directions passed inspection/targeted tests. The prior sampler used only foreground-centered patches; zero-foreground background patches are available.

## Interrupted single-patch diagnostic

- A 60-step MPS single-patch overfit attempt ran with a zero-kernel, training-patch-prior output head and foreground/background-balanced BCE plus soft Dice.
- The final disk guard detected less than 5 GiB free before results or a checkpoint could be saved. The overfit outcome is therefore unverified. No output directory/checkpoint was created.
- Latest observed free storage was about 4.83 GiB. Do not train or write more research artifacts until storage is safely above the 5 GiB floor; target at least 5.5 GiB before retrying.
- The current command implementation checks disk space after its optimization loop. Before retrying, make the single-patch runner persist progress/checkpoints periodically and check disk headroom during the loop.

## Next actions

1. Recheck disk free space; do not delete or move existing research artifacts.
2. Add periodic checkpointing and an early disk-space guard to the single-patch runner.
3. Rerun the 60-step test on one training subject only. Save initial/final foreground and background probabilities, per-channel Dice, loss history, and a separate checkpoint; verify reload.
4. Do not start another multi-epoch run until the single-patch test shows foreground/background separation.
5. If that test succeeds, consider a short cohort run with deterministic prior-bias initialization, balanced per-channel BCE plus Dice, and some background-dominant training crops. Keep validation and evaluation subjects patient-disjoint; never access the five final evaluation subjects during tuning.

## Verification and Git

- Eight focused synthetic adapter tests passed.
- The full suite/build was not run for this investigation.
- No experiment checkpoint or dataset is to be committed. `.env` is local and must remain untracked.
- Push order requested by the user: commit/push to `develop`, then merge `develop` into `master` and push `master`.
