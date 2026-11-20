from __future__ import annotations

from pathlib import Path
from random import Random
from shutil import copy2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "smartrecycle-cardboard-data"


def write_cardboard_split(source_split: str, target_split: str, negative_limit: int | None = None) -> tuple[int, int]:
    image_dir = SOURCE_ROOT / source_split / "images" / "Garbage_detection"
    label_dir = SOURCE_ROOT / source_split / "labels" / "Garbage_detection"
    images = sorted(image_dir.glob("*.jpg"))
    positives = [image for image in images if image.stem.casefold().startswith("cardboard")]
    negatives = [image for image in images if image not in positives]

    negative_limit = negative_limit if negative_limit is not None else len(positives)
    if negative_limit:
        Random(23).shuffle(negatives)
        negatives = negatives[:negative_limit]

    target_images = OUTPUT_ROOT / "images" / target_split
    target_labels = OUTPUT_ROOT / "labels" / target_split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    for image in positives:
        copy2(image, target_images / image.name)
        source_label = label_dir / f"{image.stem}.txt"
        target_rows = []
        if source_label.exists():
            for row in source_label.read_text(errors="ignore").splitlines():
                parts = row.split()
                if len(parts) == 5:
                    parts[0] = "0"
                    target_rows.append(" ".join(parts))
                elif len(parts) > 5:
                    coordinates = [float(value) for value in parts[1:]]
                    x_values, y_values = coordinates[0::2], coordinates[1::2]
                    target_rows.append(" ".join(map(str, (
                        0,
                        (min(x_values) + max(x_values)) / 2,
                        (min(y_values) + max(y_values)) / 2,
                        max(x_values) - min(x_values),
                        max(y_values) - min(y_values),
                    ))))
        (target_labels / f"{image.stem}.txt").write_text("\n".join(target_rows))

    for image in negatives:
        copy2(image, target_images / image.name)
        (target_labels / f"{image.stem}.txt").write_text("")

    return len(positives), len(negatives)


def main() -> None:
    train_counts = write_cardboard_split("train", "train")
    validation_counts = write_cardboard_split("valid", "val", negative_limit=80)
    data_file = OUTPUT_ROOT / "cardboard.yaml"
    data_file.write_text(
        f"path: {OUTPUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: Cardboard\n"
    )
    print(f"Training: {train_counts[0]} cardboard, {train_counts[1]} background images")
    print(f"Validation: {validation_counts[0]} cardboard, {validation_counts[1]} background images")
    print(f"Dataset config: {data_file}")


if __name__ == "__main__":
    main()
