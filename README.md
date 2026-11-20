# SmartRecycle

SmartRecycle is an AI-assisted waste detection and recycling guidance project. It combines a Next.js web application with an Ultralytics YOLO11 inference workflow to locate supported waste items in images and present classification results alongside disposal guidance.

> **Project scope:** SmartRecycle is a project/demo, not a certified recycling authority. Model results and the accompanying guidance should be checked against local waste rules.

## Overview

Waste is often sorted incorrectly when materials look similar, are mixed, or are contaminated. SmartRecycle explores how computer vision can make an initial waste classification easier to access and understand. A user can submit an image, review detected objects and confidence values, and consult material and disposal guidance.

### Objectives

- Provide a simple image-based waste analysis experience.
- Detect and classify items covered by the available model checkpoints.
- Show confidence, detection boxes, and label-based recycling guidance.
- Record predictions and user feedback for review.
- Provide dashboard, history, analytics, and review screens around the detection workflow.

### Key features

- Upload or drag and drop an image; the analyzer also offers camera capture.
- View detected objects, bounding boxes, confidence, and recycling metadata.
- Save prediction records and feedback to a local SQLite database.
- Review prediction history and dashboard/analytics views.
- Inspect YOLO training, validation, prediction, and export scripts.

Some screens and API responses are demo-level. In particular, pickup requests are not persisted, analytics contain fallback/demo values when there is no data, and feedback does not automatically retrain a model. See [Limitations](#limitations).

## Technology stack

| Area | Technologies |
|---|---|
| Web application | Next.js 16 App Router, React 19, TypeScript |
| Styling and UI | Tailwind CSS 4, Radix UI Slot, Lucide React, React Dropzone, React Hook Form, Sonner |
| Charts and data fetching | Recharts, TanStack React Query |
| API | Next.js route handlers and server-side TypeScript |
| Database | SQLite, Prisma 7, `better-sqlite3` adapter |
| Computer vision | Ultralytics YOLO11, Python |
| Data and model tools | Python scripts under `scripts/` and `training/`; Git LFS for model and dataset files |

## Architecture

```text
Browser (Next.js / React)
  ├── Analyzer UI
  │     └── POST /api/predict or POST /api/analyze
  │           └── TypeScript YOLO adapter
  │                 └── training/predict.py (Python / Ultralytics)
  │                       ├── Primary waste checkpoint
  │                       ├── Optional cardboard detector
  │                       └── Optional food-label detector
  └── History, feedback, and dashboard API routes
        └── Prisma 7 ── SQLite (PredictionLog and Feedback)
```

The application invokes Python inference as a child process. The prediction endpoint stores the resulting prediction in SQLite; feedback records are stored separately. The current schema has no user or pickup-request model.

### API routes

| Route | Purpose |
|---|---|
| `POST /api/predict` | Runs image inference and stores a prediction record. |
| `POST /api/analyze` | Runs image inference and returns detections without adding a prediction log record. |
| `GET /api/history` | Returns prediction history; supports search, filter, sort, and limit query parameters. |
| `GET /api/categories` | Returns the application's recycling-guidance category catalog. |
| `GET /api/statistics` | Returns dashboard/analytics aggregates and fallback values. |
| `GET /api/feedback`, `POST /api/feedback` | Reads and stores feedback records. |
| `POST /api/pickup` | Returns a pending request-shaped response; it does not persist or schedule a pickup. |

The API routes do not implement server-side authentication or authorization.

## YOLO11 workflow

### Main dataset and classes

The primary YOLO configuration is [`dataset/dataset.yaml`](dataset/dataset.yaml). It uses the `images/train`, `images/val`, and `images/test` splits and the corresponding label splits. Some source data is grouped in subdirectories under these splits. Labels use the standard YOLO text format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

The configured primary detector has **six classes**:

1. Plastic Bottle
2. Plastic Bag
3. PET Plastic
4. HDPE Plastic
5. Glass Bottle
6. Paper

The UI's category catalog is broader than the main detector's training classes; a category appearing in the interface does not mean the model has been trained to recognize it.

### Inference paths

- The application prefers `models/waste-best.pt` as its primary waste-trained checkpoint and runs the inference code in `training/predict.py`. The application uses a primary confidence threshold of `0.60`.
- If the primary model returns no detections or has detections below the configured confidence condition, it can also run `models/cardboard-best.pt` at `0.60`. This is a separate one-class cardboard model.
- Under the same uncertainty condition, `models/best.pt` can be used as an auxiliary detector for explicitly recognized common food labels. Those labels are reported as Food Waste; unrelated generic objects are not assigned a guessed waste class.
- Objects not mapped to a supported waste label are either omitted or returned as unclassified, depending on the inference path. A no-detection result is presented for manual identification rather than inventing a material.

For the focused cardboard training data, [`training/prepare_cardboard_dataset.py`](training/prepare_cardboard_dataset.py) selects images whose names begin with `cardboard` as positives from the source `Garbage_detection` set and samples other images as backgrounds. It writes its generated dataset under `runs/smartrecycle-cardboard-data/`.

### Training and evaluation

[`training/train.py`](training/train.py) starts from `yolo11n.pt` by default, uses `dataset/dataset.yaml`, and defaults to 50 epochs, image size 640, and batch size 16. A recorded run is described in [`docs/model-evaluation.md`](docs/model-evaluation.md); it used 10 epochs and batch size 8. Training outputs are written under `training/` by default. Training does not automatically replace the application's `models/waste-best.pt` checkpoint.

The repository includes training, validation, test, prediction, evaluation, and export scripts under `training/`. Use `--help` on a script for its available options. For example:

```bash
# Train using the script defaults, overriding selected options
npm run yolo:train -- --epochs 10 --batch 8

# Validate the primary checkpoint against the configured validation split
npm run yolo:validate -- --weights models/waste-best.pt

# Evaluate the primary checkpoint on the test split
npm run yolo:test -- --weights models/waste-best.pt --task test

# Predict on one image
npm run yolo:predict -- --weights models/waste-best.pt --image path/to/image.jpg

# Export directly with the script's supported CLI
python training/export.py --weights models/waste-best.pt --format onnx
```

The Python scripts require Ultralytics. To use a newly trained checkpoint in the web application, copy or move it to `models/waste-best.pt`. The application also needs a working Python interpreter and the applicable model files at inference time.

## Model evaluation and screenshots

The repository's [model evaluation notes](docs/model-evaluation.md) report results for the separate cardboard detector on a 156-image held-out validation split at confidence `0.60`:

| Metric | Recorded value |
|---|---:|
| Precision | 0.841 |
| Recall | 0.853 |
| mAP50 | 0.783 |
| mAP50-95 | 0.688 |

These are the documented results for that checkpoint and split only; they are not an overall SmartRecycle accuracy score or a guarantee of performance on other images. No overall final accuracy score is claimed.

The tracked model figures are experiment artifacts, not product screenshots:

- [YOLO training results](docs/screenshots/yolo-training-results.png)
- [Confusion matrix](docs/screenshots/yolo-confusion-matrix.png)
- [Normalized confusion matrix](docs/screenshots/yolo-confusion-matrix-normalized.png)
- [Validation labels](docs/screenshots/yolo-validation-labels.jpg)
- [Validation predictions](docs/screenshots/yolo-validation-predictions.jpg)

No application UI screenshots are currently included in `docs/screenshots/`.

## Project structure

```text
SmartRecycle/
├── dataset/                 # YOLO dataset configuration, images, labels, and notes
├── docs/                    # Model evaluation notes and YOLO figures
├── models/                  # Inference checkpoints and exports (Git LFS)
├── prisma/                  # SQLite schema and migrations
├── public/                  # Static assets
├── scripts/                 # Dataset preparation, validation, and statistics
├── src/
│   ├── app/                 # Next.js pages and API route handlers
│   ├── components/          # Analyzer, dashboard, admin, pickup, and UI
│   ├── hooks/               # Client-side hooks
│   ├── lib/                 # Auth, Prisma, inference, data, and utilities
│   └── generated/prisma/    # Generated Prisma client
├── training/                # YOLO training, prediction, evaluation, and export
├── .env.example             # Environment variable template
├── package.json             # npm scripts and JavaScript dependencies
├── prisma.config.ts         # Prisma schema, migration, and datasource config
└── README.md
```

## Setup

### Prerequisites

- Node.js `>=20.9.0` and npm (the Next.js 16 package declares Node.js `>=20.9.0`).
- Python 3 and `pip`. Python is used by the YOLO training and inference scripts.
- Git LFS to download the tracked dataset and model artifacts.
- Sufficient disk space for the LFS-managed images and model weights.

The repository does not include a Python dependency lockfile or `requirements.txt`; the Python scripts import Ultralytics. Install it into the interpreter that will run inference and training.

### Clone and install JavaScript dependencies

```bash
git lfs install
git clone https://github.com/buildwithanish2/SmartRecycle.git
cd SmartRecycle
npm install
```

If Git LFS was not installed when cloning, run `git lfs pull` from the repository directory to fetch the dataset and model files.

### Configure Python for YOLO

From PowerShell on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics
```

The application automatically uses `.venv\Scripts\python.exe` when that file exists; otherwise it runs `python`. Set `PYTHON_BIN` in `.env` if you need to choose a different interpreter.

### Configure environment

Create `.env` from the checked-in template.

PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

| Variable | Template default | Purpose |
|---|---|---|
| `DATABASE_URL` | `file:./dev.db` | SQLite URL used by Prisma. |
| `PYTHON_BIN` | Empty | Optional Python executable override for the active YOLO inference adapter. |
| `WASTE_MODEL_PATH` | Empty | Read by the separate `src/lib/inference-service.ts` adapter; the active `/api/predict` route uses `src/lib/yolo-model.ts` and does not use this variable to select its checkpoint. |

Do not commit `.env`; it is ignored by Git.

### Initialize the database

The Prisma schema is in [`prisma/schema.prisma`](prisma/schema.prisma). It defines `PredictionLog` and `Feedback` models on SQLite.

```bash
npx prisma generate
npx prisma migrate dev
```

### Run the web application

```bash
npm run dev
```

Open the local URL printed by Next.js (normally `http://localhost:3000`). The root page opens the public analyzer.

To create and run a production build:

```bash
npm run build
npm run start
```

Other package scripts include `npm run lint`, `npm run dataset:validate`, and `npm run dataset:stats`. The exact scripts are listed in [`package.json`](package.json).

## Using the application

1. Open the public analyzer.
2. Upload, drag and drop, or capture an image.
3. Review any returned detections and bounding boxes.
4. Check the predicted label, confidence, material metadata, and recycling recommendation.
5. Submit feedback if the result is incorrect.
6. Open the dashboard to review logged predictions, history, and available analytics/review screens.

The guidance is an informational aid. Local recycling rules vary, and the analyzer cannot reliably assess contamination or confirm material composition from every image.

## Authentication: current behavior

The login screen asks for an email and password, but it does not validate credentials or contact an authentication service. Submitting any non-empty values sets `smartrecycle-authenticated=true` in browser `localStorage`; the dashboard checks that flag in the client and redirects to the login page if it is absent. Logging out clears the browser-side flag and returns to the home page.

This is a demo navigation gate only. It is not secure authentication, does not create user accounts, and does not protect API routes or database records.

## Limitations and future improvements

- The primary trained dataset has six classes. Cardboard and common food labels use auxiliary detectors; other guidance categories are not necessarily recognized by a trained model.
- Detection confidence is not a calibrated guarantee. The material, contamination, and disposal text is derived from label-based application rules; it is not a separate verified contamination analysis.
- Image quality, lighting, occlusion, mixed materials, and local recycling practices can affect whether a recommendation is appropriate.
- The client-only login is not an identity or authorization system.
- Feedback is stored for review but is not converted into annotations or used for automatic retraining.
- Analytics can return fallback values when records are absent; treat these as demo UI data, not model evaluation results.
- Pickup submission currently returns a pending response without persistence or integration with a collection service.

Possible next steps include expanding and documenting the labeled dataset, defining a repeatable evaluation protocol, building a reviewed feedback-to-training pipeline, implementing server-side authentication and authorization, persisting pickup requests, and replacing demo analytics defaults with explicitly sourced values.
