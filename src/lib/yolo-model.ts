import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readdir, stat, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { BoundingBox, PredictionObject, PredictionResponse } from "@/lib/types";

function getPythonBin(): string {
  if (process.env.PYTHON_BIN) {
    return process.env.PYTHON_BIN;
  }
  const venvPython = path.resolve(process.cwd(), ".venv", "Scripts", "python.exe");
  if (existsSync(venvPython)) {
    return venvPython;
  }
  return "python";
}

const PYTHON_BIN = getPythonBin();
const MODEL_ROOT = path.resolve(process.cwd(), "models");
const WEIGHTS_ROOT = path.resolve(process.cwd(), "weights");
const CARDBOARD_MODEL_PATH = path.join(MODEL_ROOT, "cardboard-best.pt");
const COCO_MODEL_PATH = path.join(MODEL_ROOT, "best.pt");
const INFERENCE_SCRIPT = path.resolve(process.cwd(), "training", "predict.py");

function toDetString(value: string) {
  return value.replace(/[^a-zA-Z0-9._-]/g, "-").toLowerCase();
}

async function ensureModelDirectory() {
  await mkdir(MODEL_ROOT, { recursive: true });
  await mkdir(WEIGHTS_ROOT, { recursive: true });
}

async function getNewestModelPath() {
  await ensureModelDirectory();

  // Prefer the waste-trained checkpoint over the older generic COCO exports.
  const wasteModelPath = path.join(MODEL_ROOT, "waste-best.pt");
  if (existsSync(wasteModelPath)) {
    return wasteModelPath;
  }

  const candidates = [
    ...await listFiles(MODEL_ROOT),
    ...await listFiles(WEIGHTS_ROOT),
    ...(existsSync(path.resolve(process.cwd(), "yolo11n.pt"))
      ? [{ fullPath: path.resolve(process.cwd(), "yolo11n.pt"), mtimeMs: 1 }]
      : []),
  ]
    .filter((candidate) => candidate.fullPath.endsWith(".onnx") || candidate.fullPath.endsWith(".tflite") || candidate.fullPath.endsWith(".pt"))
    .sort((left, right) =>
      right.mtimeMs - left.mtimeMs ||
      Number(right.fullPath.endsWith(".pt")) - Number(left.fullPath.endsWith(".pt")),
    );

  return candidates[0]?.fullPath ?? null;
}

async function listFiles(directory: string) {
  try {
    const entries = await readdir(directory, { withFileTypes: true });
    const files: Array<{ fullPath: string; mtimeMs: number }> = [];

    for (const entry of entries) {
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        files.push(...await listFiles(fullPath));
        continue;
      }

      const fileStats = await stat(fullPath);
      files.push({ fullPath, mtimeMs: fileStats.mtimeMs });
    }

    return files;
  } catch {
    return [];
  }
}

interface EnrichedMetadata {
  material: string;
  recyclable: boolean;
  contamination: "Low" | "Medium" | "High";
  cleaning: string;
  recommendation: string;
  explanation: string[];
}

function getWasteMetadata(label: string, confidence: number): EnrichedMetadata {
  const norm = label.toLowerCase();

  if (norm.startsWith("unclassified")) {
    return {
      material: "Unknown Material",
      recyclable: false,
      contamination: "High",
      cleaning: "Keep dry and do not dismantle the item until its material is identified.",
      recommendation: "Identify the item first. If it may be a battery or electronic device, use a dedicated e-waste drop-off; otherwise check local sorting rules.",
      explanation: ["An object was detected, but its waste material was not confirmed", "No recycling category was guessed"],
    };
  }

  if (norm.includes("glass") || norm.includes("wine glass")) {
    return {
      material: "Glass",
      recyclable: true,
      contamination: "Low",
      cleaning: "Rinse clean; remove non-glass caps.",
      recommendation: "Recycle in Glass Bin",
      explanation: ["Transparent / colored vitreous body", "Smooth rigid glass surface contour"],
    };
  }
  if (norm.includes("plastic bag") || norm.includes("bag") || norm.includes("film")) {
    return {
      material: "LDPE Soft Plastic",
      recyclable: true,
      contamination: "Medium",
      cleaning: "Ensure bag is clean and free of food scraps.",
      recommendation: "Recycle at Soft Plastic Drop-Off",
      explanation: ["Flexible polymer film texture", "Pliable lightweight membrane"],
    };
  }
  if (norm.includes("hdpe")) {
    return {
      material: "HDPE Plastic",
      recyclable: true,
      contamination: "Low",
      cleaning: "Rinse bottle thoroughly and replace lid.",
      recommendation: "Recycle in Plastics Bin",
      explanation: ["Opaque rigid plastic container", "High-density polyethylene surface"],
    };
  }
  if (norm.includes("plastic bottle") || norm.includes("bottle") || norm.includes("pet")) {
    return {
      material: "PET Plastic",
      recyclable: true,
      contamination: confidence > 80 ? "Low" : "Medium",
      cleaning: "Rinse and empty remaining liquids; keep cap on if accepted locally.",
      recommendation: "Recycle in Plastics Bin",
      explanation: ["Transparent / shaped thermoplastic profile", "Cylindrical bottle neck geometry", "PET resin characteristics detected"],
    };
  }
  if (norm.includes("cardboard") || norm.includes("box")) {
    return {
      material: "Corrugated Cardboard",
      recyclable: true,
      contamination: "Low",
      cleaning: "Flatten and keep dry; remove tape if excessive.",
      recommendation: "Recycle in Paper/Cardboard Bin",
      explanation: ["Multi-layer corrugated structure", "Kraft paper pulp texture"],
    };
  }
  if (norm.includes("paper") || norm.includes("book")) {
    return {
      material: "Paper Fibre",
      recyclable: true,
      contamination: "Low",
      cleaning: "Keep dry and free from oil or grease.",
      recommendation: "Recycle in Paper Bin",
      explanation: ["Flat cellulose paper surface", "Lightweight fiber composition"],
    };
  }
  if (norm.includes("aluminium") || norm.includes("can") || norm.includes("tin")) {
    return {
      material: "Aluminium / Metal",
      recyclable: true,
      contamination: "Medium",
      cleaning: "Rinse residue and crush if preferred.",
      recommendation: "Recycle in Metal Bin",
      explanation: ["Metallic reflective finish", "Cylindrical beverage can profile", "High recyclable scrap value"],
    };
  }
  if (norm.includes("food") || norm.includes("organic") || norm.includes("banana") || norm.includes("apple") || norm.includes("sandwich") || norm.includes("pizza")) {
    return {
      material: "Organic Matter",
      recyclable: false,
      contamination: "High",
      cleaning: "Compost directly; do not place in dry recycling.",
      recommendation: "Dispose in Compost / Organic Bin",
      explanation: ["Perishable food substance", "Biological matter unsuitable for dry recycling"],
    };
  }
  if (norm.includes("battery")) {
    return {
      material: "Lithium / Chemical Cell",
      recyclable: false,
      contamination: "High",
      cleaning: "Do not wet or crush; tape terminals.",
      recommendation: "Take to Dedicated Battery Drop-Off",
      explanation: ["Electrochemical cell casing", "Hazardous material requiring dedicated handling"],
    };
  }
  if (norm.includes("e-waste") || norm.includes("phone") || norm.includes("laptop") || norm.includes("mouse") || norm.includes("keyboard")) {
    return {
      material: "Electronic Waste",
      recyclable: false,
      contamination: "High",
      cleaning: "Do not dismantle; keep dry.",
      recommendation: "Take to E-Waste Recycling Point",
      explanation: ["Circuitry and electronic component detected", "Requires specialized dismantling"],
    };
  }
  if (norm.includes("textile") || norm.includes("cloth")) {
    return {
      material: "Fabric / Textile",
      recyclable: true,
      contamination: "Medium",
      cleaning: "Wash and dry before donation or recycling.",
      recommendation: "Deposit in Textile Collection Bin",
      explanation: ["Woven / knitted fabric structure", "Reclaimable textile fibers"],
    };
  }
  if (norm.includes("tetra")) {
    return {
      material: "Composite Carton",
      recyclable: true,
      contamination: "Medium",
      cleaning: "Rinse and flatten.",
      recommendation: "Recycle in Carton / Mixed Stream",
      explanation: ["Layered paper-foil-polyethylene packaging"],
    };
  }

  return {
    material: label || "Mixed Recyclable",
    recyclable: true,
    contamination: confidence > 75 ? "Low" : "Medium",
    cleaning: "Rinse and clean before disposal.",
    recommendation: "Check local municipal recycling rules.",
    explanation: ["Visual features consistent with recyclable packaging", "Automated detection match"],
  };
}

function runPythonInference(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, args, {
      cwd: process.cwd(),
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let settled = false;

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => { stdout += chunk; });
    child.stderr.on("data", (chunk: string) => { stderr += chunk; });

    const timeout = setTimeout(() => {
      child.kill();
      settled = true;
      reject(new Error("YOLO inference timed out. Try again with a smaller image."));
    }, 120_000);

    child.once("error", (error) => {
      clearTimeout(timeout);
      if (settled) return;
      settled = true;
      reject(new Error(`Could not start Python inference: ${error.message}`));
    });

    child.once("close", (code) => {
      clearTimeout(timeout);
      if (settled) return;
      settled = true;
      if (code !== 0) {
        const detail = stderr.trim().split("\n").slice(-3).join(" ");
        reject(new Error(detail || "YOLO inference failed. Check the Python environment and model files."));
        return;
      }
      resolve(stdout);
    });
  });
}

export async function inferWasteWithYolo(file: File): Promise<PredictionResponse> {
  const modelPath = await getNewestModelPath();
  if (!modelPath) {
    throw new Error("No trained YOLO model export found. Train or export a model before running inference.");
  }

  const imageBuffer = Buffer.from(await file.arrayBuffer());
  const extensionByType: Record<string, string> = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
  };
  const extension = extensionByType[file.type] ?? ".jpg";
  const tempFilePath = path.join(tmpdir(), `${toDetString(path.basename(file.name || "waste-image", path.extname(file.name || "")))}-${crypto.randomUUID()}${extension}`);
  await writeFile(tempFilePath, imageBuffer);

  let rawObjects: Array<{
    label: string;
    confidence: number;
    boundingBox: BoundingBox & { x1?: number; y1?: number; x2?: number; y2?: number };
  }> = [];

  try {
    const inferenceArgs = [
      INFERENCE_SCRIPT,
      "--weights",
      modelPath,
      "--image",
      tempFilePath,
      "--conf",
      "0.6",
    ];
    if (existsSync(CARDBOARD_MODEL_PATH)) {
      inferenceArgs.push("--cardboard-weights", CARDBOARD_MODEL_PATH, "--cardboard-conf", "0.6");
    }
    if (existsSync(COCO_MODEL_PATH)) {
      inferenceArgs.push("--food-weights", COCO_MODEL_PATH, "--food-conf", "0.4");
    }

    const stdout = (await runPythonInference(inferenceArgs)).trim();
    const jsonStart = stdout.indexOf("{");
    const jsonEnd = stdout.lastIndexOf("}");
    if (jsonStart === -1 || jsonEnd === -1) {
      throw new Error("YOLO inference returned an invalid response.");
    }

    const parsed = JSON.parse(stdout.substring(jsonStart, jsonEnd + 1)) as { objects?: typeof rawObjects };
    if (!Array.isArray(parsed.objects)) {
      throw new Error("YOLO inference returned an invalid detections list.");
    }
    rawObjects = parsed.objects;
  } finally {
    await unlink(tempFilePath).catch(() => undefined);
  }

  if (rawObjects.length === 0) {
    const explanation = [
      "The image was received, but no object could be identified confidently.",
      "No waste type or material was guessed; please identify it manually.",
    ];
    const recommendation = "Identify the item first. If it may be a battery or electronic device, use a dedicated e-waste drop-off; otherwise check local sorting rules.";

    return {
      id: crypto.randomUUID(),
      prediction: "Unclassified Item",
      confidence: 0,
      material: "Unknown Material",
      recyclable: false,
      contamination: "High",
      cleaning: "Keep dry and do not dismantle the item until its material is identified.",
      recommendation,
      explanation,
      alternatives: [],
      status: "completed",
      detections: [],
      objects: [],
    };
  }

  const enrichedObjects: PredictionObject[] = rawObjects.map((raw) => {
    const meta = getWasteMetadata(raw.label, raw.confidence);
    return {
      label: raw.label,
      confidence: Number(raw.confidence.toFixed(1)),
      boundingBox: {
        x: raw.boundingBox.x ?? raw.boundingBox.x1 ?? 0,
        y: raw.boundingBox.y ?? raw.boundingBox.y1 ?? 0,
        width: raw.boundingBox.width ?? Math.max(10, (raw.boundingBox.x2 ?? 100) - (raw.boundingBox.x1 ?? 0)),
        height: raw.boundingBox.height ?? Math.max(10, (raw.boundingBox.y2 ?? 100) - (raw.boundingBox.y1 ?? 0)),
      },
      material: meta.material,
      recyclable: meta.recyclable,
      contamination: meta.contamination,
      recommendation: meta.recommendation,
      explanation: meta.explanation,
    };
  });

  const primary = enrichedObjects[0];
  const primaryMeta = getWasteMetadata(primary.label, primary.confidence);

  return {
    id: crypto.randomUUID(),
    prediction: primary.label,
    confidence: primary.confidence,
    material: primary.material,
    recyclable: primary.recyclable,
    contamination: primary.contamination,
    cleaning: primaryMeta.cleaning,
    recommendation: primary.recommendation,
    explanation: primary.explanation,
    alternatives: enrichedObjects.slice(0, 3).map((object) => ({
      label: object.label,
      confidence: Number(object.confidence.toFixed(1)),
    })),
    status: "completed",
    detections: enrichedObjects.map((object) => {
      const meta = getWasteMetadata(object.label, object.confidence);
      return {
        type: object.label,
        material: object.material,
        confidence: Number(object.confidence.toFixed(1)),
        recyclable: object.recyclable,
        contamination: object.contamination,
        cleaning: meta.cleaning,
        recommendation: object.recommendation,
        explanation: object.explanation,
        alternatives: enrichedObjects.slice(0, 3).map((candidate) => ({
          label: candidate.label,
          confidence: Number(candidate.confidence.toFixed(1)),
        })),
        boundingBox: object.boundingBox,
        label: object.label,
      };
    }),
    objects: enrichedObjects,
  };
}

export async function getLatestInferenceModelPath() {
  return getNewestModelPath();
}

