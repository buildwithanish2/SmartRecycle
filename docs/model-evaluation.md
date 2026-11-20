# Model Evaluation Notes

The project contains a recorded Ultralytics YOLO11 training run under the local `runs/` directory. The run configuration records a detection task using `yolo11n.pt`, the project dataset, 10 epochs, and batch size 8.

The local `results.csv` includes epoch-level box loss, classification loss, DFL loss, precision, recall, mAP50, and mAP50-95 columns. It is generated training output and is intentionally not committed to the GitHub repository with the raw `runs/` tree.

The repository does not publish a single verified final accuracy number because the available output is an epoch history and the project does not define a release evaluation protocol. The original project contains validation plots and confusion matrices, which are copied into `docs/screenshots/` when present.

**Evaluation metrics are pending/future work as a formally reported project result.** The available local plots should be treated as experiment artifacts, not as a generalization guarantee.

## Cardboard Detector

The application also includes a focused one-class cardboard detector at `models/cardboard-best.pt`. Its training set was prepared from the dataset's `Garbage_detection` source using images explicitly named `cardboard*` as positives and other images from that source as backgrounds. The repeatable preparation script is `training/prepare_cardboard_dataset.py`.

On the held-out 156-image validation split at confidence 0.60, the checkpoint measured precision 0.841, recall 0.853, mAP50 0.783, and mAP50-95 0.688. The user's attached cardboard screenshot crop also produced a Cardboard detection at 80.1% confidence. These results are specific to this small validation set and do not guarantee the same accuracy for every cardboard shape or scene.
