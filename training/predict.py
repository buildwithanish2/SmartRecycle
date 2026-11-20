from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from ultralytics import YOLO

SUPPORTED_WASTE_CLASSES = {
    "plastic bottle": "Plastic Bottle",
    "plastic bag": "Plastic Bag",
    "pet plastic": "PET Plastic",
    "hdpe plastic": "HDPE Plastic",
    "glass bottle": "Glass Bottle",
    "paper": "Paper",
    "cardboard": "Cardboard",
    "aluminium can": "Aluminium Can",
    "steel can": "Steel Can",
    "food waste": "Food Waste",
    "organic waste": "Organic Waste",
    "e-waste": "E-Waste",
    "battery": "Battery",
    "textile": "Textile",
    "tetra pak": "Tetra Pak",
    "others": "Others",
}
COCO_FOOD_CLASSES = {
    "banana": "Food Waste",
    "apple": "Food Waste",
    "sandwich": "Food Waste",
    "orange": "Food Waste",
    "broccoli": "Food Waste",
    "carrot": "Food Waste",
    "hot dog": "Food Waste",
    "pizza": "Food Waste",
    "donut": "Food Waste",
    "cake": "Food Waste",
}


def normalize_waste_label(raw_label: str) -> str | None:
    return SUPPORTED_WASTE_CLASSES.get(raw_label.strip().casefold())


def collect_detections(result, label_aliases: dict[str, str] | None = None) -> list[dict]:
    detections = []
    for box in result.boxes:
        cls_idx = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
        cls_name = result.names.get(cls_idx, f"Class {cls_idx}") if isinstance(result.names, dict) else result.names[cls_idx]
        conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
        xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, "__len__") else list(box.xyxy)
        x1, y1, x2, y2 = [float(val) for val in xyxy]

        normalized_label = normalize_waste_label(str(cls_name))
        if normalized_label is None and label_aliases:
            normalized_label = label_aliases.get(str(cls_name).strip().casefold())
        if normalized_label is None:
            continue

        detections.append({
            "label": normalized_label,
            "confidence": round(conf * 100, 1) if conf <= 1.0 else round(conf, 1),
            "boundingBox": {
                "x": round(x1, 2),
                "y": round(y1, 2),
                "width": round(max(0.0, x2 - x1), 2),
                "height": round(max(0.0, y2 - y1), 2),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
        })
    return detections


def collect_generic_detections(result, min_confidence: float = 0.25) -> list[dict]:
    detections = []
    for box in result.boxes:
        cls_idx = int(box.cls[0])
        class_name = result.names.get(cls_idx, f"Class {cls_idx}") if isinstance(result.names, dict) else result.names[cls_idx]
        raw_label = str(class_name).strip().casefold()
        confidence = float(box.conf[0])
        if confidence < min_confidence or raw_label in COCO_FOOD_CLASSES:
            continue

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        detections.append({
            "label": f"Unclassified Object: {str(class_name).strip().title()}",
            "confidence": round(confidence * 100, 1),
            "boundingBox": {
                "x": round(x1, 2),
                "y": round(y1, 2),
                "width": round(max(0.0, x2 - x1), 2),
                "height": round(max(0.0, y2 - y1), 2),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
        })
    return detections


def box_iou(first: dict, second: dict) -> float:
    a = first["boundingBox"]
    b = second["boundingBox"]
    intersection_width = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    intersection_height = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    intersection = intersection_width * intersection_height
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image")
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--source", "--image", dest="source", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--cardboard-weights", type=Path)
    parser.add_argument("--cardboard-conf", type=float, default=0.6)
    parser.add_argument("--food-weights", type=Path)
    parser.add_argument("--food-conf", type=float, default=0.4)
    parser.add_argument("--generic-conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    try:
        model = YOLO(str(args.weights))
        model_names = model.names.values() if isinstance(model.names, dict) else model.names
        if not any(normalize_waste_label(str(name)) for name in model_names):
            raise RuntimeError(
                "The selected YOLO weights do not contain SmartRecycle waste classes. "
                "Use weights trained with the waste dataset before analyzing images."
            )

        result = model(args.source, conf=args.conf, save=args.save, verbose=False)[0]
        detections = collect_detections(result)

        # The main dataset does not contain a reliable cardboard class. Run a
        # dedicated detector when the main model is uncertain or finds nothing.
        if args.cardboard_weights and (
            not detections or any(item["confidence"] < 75 for item in detections)
        ):
            cardboard_model = YOLO(str(args.cardboard_weights))
            cardboard_names = cardboard_model.names.values() if isinstance(cardboard_model.names, dict) else cardboard_model.names
            if not any(str(name).strip().casefold() == "cardboard" for name in cardboard_names):
                raise RuntimeError("The configured cardboard detector does not contain the Cardboard class.")

            cardboard_result = cardboard_model(
                args.source,
                conf=args.cardboard_conf,
                verbose=False,
            )[0]
            cardboard_detections = [
                item for item in collect_detections(cardboard_result)
                if item["label"] == "Cardboard"
            ]
            for cardboard_detection in cardboard_detections:
                detections = [
                    item for item in detections
                    if box_iou(item, cardboard_detection) < 0.3
                ]
                detections.append(cardboard_detection)

        # COCO has reliable labels for common foods but not waste materials.
        # Restrict this auxiliary model to explicit food classes only.
        if args.food_weights and (
            not detections or any(item["confidence"] < 75 for item in detections)
        ):
            food_model = YOLO(str(args.food_weights))
            food_names = food_model.names.values() if isinstance(food_model.names, dict) else food_model.names
            if any(str(name).strip().casefold() in COCO_FOOD_CLASSES for name in food_names):
                food_result = food_model(
                    args.source,
                    conf=min(args.food_conf, args.generic_conf),
                    verbose=False,
                )[0]
                food_detections = [
                    item for item in collect_detections(food_result, COCO_FOOD_CLASSES)
                    if item["label"] == "Food Waste" and item["confidence"] >= args.food_conf * 100
                ]
                for food_detection in food_detections:
                    detections = [
                        item for item in detections
                        if box_iou(item, food_detection) < 0.3
                    ]
                    detections.append(food_detection)

                if not detections:
                    detections = collect_generic_detections(food_result, args.generic_conf)
    except Exception as error:
        print(f"YOLO inference failed: {error}", file=sys.stderr)
        raise

    print(json.dumps({"objects": detections}, indent=2))


if __name__ == "__main__":
    main()
