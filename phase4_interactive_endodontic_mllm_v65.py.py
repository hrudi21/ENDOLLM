#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

# Frozen vision pipeline lives in EndoMLLM root.
sys.path.insert(0, str(Path.home() / "EndoMLLM"))
from diagnostic_rules import evaluate_diagnostic_consistency
from non_endodontic_red_flags import assess_non_endodontic_red_flags
from vision_pipeline import analyze_radiograph


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

KB = Path.home() / "EndoMLLM/phase2_knowledge"

EMBED_FILE = KB / "indexes/bge_small_embeddings.npy"
META_FILE = KB / "indexes/bge_small_metadata.jsonl"
CHUNKS_FILE = KB / "processed_chunks/chunks.jsonl"

# Merged index: original endodontic corpus + non-endodontic red-flag corpus.
MERGED_EMBED_FILE = KB / "indexes/bge_small_embeddings_merged_244.npy"
MERGED_META_FILE = KB / "indexes/bge_small_metadata_merged_244.jsonl"
MERGED_CHUNKS_FILE = KB / "processed_chunks/chunks_merged_244.jsonl"

OUTPUT_DIR = KB / "evaluation/phase4_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 2048 * 28 * 28

TOP_K = 5

# OPGAgent is an additive visual-evidence layer. The original v62
# FasterRCNN/ResNet50 pipeline remains untouched.
OPGAGENT_ROOT = Path.home() / "OPGAgent"
OPGAGENT_PYTHON = Path.home() / "opgagent_env/bin/python"
OPGAGENT_TVEM = OPGAGENT_ROOT / "api_service/tvem"
OPGAGENT_YOLO = OPGAGENT_ROOT / "api_service/yolo_enumeration"
OPGAGENT_OUTPUT_DIR = KB / "evaluation/opgagent_outputs"
OPGAGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPGAGENT_YOLO_MODEL = OPGAGENT_YOLO / "model/best.pt"
OPGAGENT_TVEM_MODELS = {
    "11diseases": {
        "config": OPGAGENT_TVEM / "configs/11diseases.yaml",
        "checkpoint": (
            OPGAGENT_TVEM
            / "weights/Teeth_Visual_Experts_Maskdino_Swinl_x-ray_11diseases.pth"
        ),
        "category_file": OPGAGENT_TVEM / "categories/11diseases_category.json",
    },
    "4quadrants": {
        "config": OPGAGENT_TVEM / "configs/4quadrants.yaml",
        "checkpoint": (
            OPGAGENT_TVEM
            / "weights/Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_4quadrants.pth"
        ),
        "category_file": OPGAGENT_TVEM / "categories/4quadrants_category.json",
    },
    "mandibular_maxillary": {
        "config": OPGAGENT_TVEM / "configs/mandibular_maxillary.yaml",
        "checkpoint": (
            OPGAGENT_TVEM
            / "weights/Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.pth"
        ),
        "category_file": (
            OPGAGENT_TVEM / "categories/mandibular_maxillary_category.json"
        ),
    },
    "bone_loss": {
        "config": OPGAGENT_TVEM / "configs/bone_loss.yaml",
        "checkpoint": (
            OPGAGENT_TVEM
            / "weights/Teeth_Visual_Experts_Maskdino_Swinl_x-ray_bone_loss_1disease.pth"
        ),
        "category_file": OPGAGENT_TVEM / "categories/bone_loss_category.json",
    },
}

TOOTH_TYPE_NAMES = {
    "1": "central incisor",
    "2": "lateral incisor",
    "3": "canine",
    "4": "first premolar",
    "5": "second premolar",
    "6": "first molar",
    "7": "second molar",
    "8": "third molar",
}

QUADRANT_DISPLAY = {
    "Upper Right": "Upper right",
    "Upper Left": "Upper left",
    "Lower Right": "Lower right",
    "Lower Left": "Lower left",
}


# ============================================================
# QWEN
# ============================================================


def load_qwen():
  print("=" * 90)
  print("PHASE 4 INTERACTIVE ENDODONTIC MLLM (v65) — STAGE 7.2")
  print("=" * 90)
  print(f"Loading base {BASE_MODEL}...")

  processor = AutoProcessor.from_pretrained(
      BASE_MODEL,
      min_pixels=MIN_PIXELS,
      max_pixels=MAX_PIXELS,
  )

  bnb = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.float16,
  )

  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      BASE_MODEL,
      quantization_config=bnb,
      device_map="auto",
      torch_dtype=torch.float16,
  )

  model.eval()

  print(
      f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
  )

  return model, processor


def qwen_text(model, processor, prompt, max_new_tokens=300):
  messages = [{
      "role": "user",
      "content": [
          {
              "type": "text",
              "text": prompt,
          }
      ],
  }]

  text = processor.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
  )

  inputs = processor(
      text=[text],
      padding=True,
      return_tensors="pt",
  )

  inputs = {
      key: value.to(model.device) if torch.is_tensor(value) else value
      for key, value in inputs.items()
  }

  with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=(
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        ),
    )

  generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

  return processor.batch_decode(
      generated_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False,
  )[0].strip()


# ============================================================
# RAG
# ============================================================


def load_rag_index(merged=False):
  print("Loading BGE-small retrieval index...")

  if merged:
    embed_file = MERGED_EMBED_FILE
    meta_file = MERGED_META_FILE
    chunks_file = MERGED_CHUNKS_FILE
  else:
    embed_file = EMBED_FILE
    meta_file = META_FILE
    chunks_file = CHUNKS_FILE

  embeddings = np.load(embed_file)

  metadata = [
      json.loads(line)
      for line in meta_file.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]

  chunks = [
      json.loads(line)
      for line in chunks_file.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]

  retriever = SentenceTransformer(
      "BAAI/bge-small-en-v1.5",
      device="cuda" if torch.cuda.is_available() else "cpu",
  )

  index_name = "MERGED-244" if merged else "ENDO-209"
  print(f"Retrieval index loaded: {index_name} | {len(chunks)} chunks")

  return retriever, embeddings, metadata, chunks


def retrieve_evidence(
    retriever,
    embeddings,
    metadata,
    chunks,
    query,
    top_k=TOP_K,
):
  q = retriever.encode(
      [query],
      normalize_embeddings=True,
      convert_to_numpy=True,
  )[0]

  scores = embeddings @ q

  top_k = min(top_k, len(scores))
  indices = np.argsort(scores)[::-1][:top_k]

  evidence = []

  for rank, idx in enumerate(indices, start=1):
    meta = metadata[idx]
    chunk = chunks[idx]

    evidence.append({
        "rank": rank,
        "score": float(scores[idx]),
        "source_file": meta["source_file"],
        "page_start": meta["page_start"],
        "page_end": meta["page_end"],
        "chunk_id": meta["chunk_id"],
        "text": chunk["text"],
    })

  return evidence


# ============================================================
# OPGAGENT ADDITIVE VISUAL EVIDENCE
# ============================================================


def _opgagent_env():
  env = os.environ.copy()
  torch_lib = Path(sys.executable).resolve().parent / "site-packages/torch/lib"
  if torch_lib.exists():
    env["LD_LIBRARY_PATH"] = (
        str(torch_lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    )
  return env


def _run_opgagent(command, cwd, label):
  print(f"\n[OPGAgent] {label}")
  result = subprocess.run(
      command,
      cwd=str(cwd),
      env=_opgagent_env(),
      text=True,
      capture_output=True,
  )
  if result.returncode != 0:
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    raise RuntimeError(
        f"OPGAgent {label} failed with exit code {result.returncode}."
    )
  if result.stdout:
    print(result.stdout.strip())
  return result


def _read_json(path):
  with Path(path).open("r", encoding="utf-8") as f:
    return json.load(f)


def _bbox_xyxy(box):
  if not box:
    return None
  if all(k in box for k in ("x1", "y1", "x2", "y2")):
    return (
        float(box["x1"]),
        float(box["y1"]),
        float(box["x2"]),
        float(box["y2"]),
    )
  if all(k in box for k in ("x", "y", "width", "height")):
    x = float(box["x"])
    y = float(box["y"])
    return (x, y, x + float(box["width"]), y + float(box["height"]))
  return None


def _iou_xyxy(a, b):
  a = _bbox_xyxy(a) if isinstance(a, dict) else a
  b = _bbox_xyxy(b) if isinstance(b, dict) else b
  if not a or not b:
    return 0.0
  ax1, ay1, ax2, ay2 = a
  bx1, by1, bx2, by2 = b
  ix1, iy1 = max(ax1, bx1), max(ay1, by1)
  ix2, iy2 = min(ax2, bx2), min(ay2, by2)
  iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
  inter = iw * ih
  if inter <= 0:
    return 0.0
  area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
  area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
  union = area_a + area_b - inter
  return inter / union if union > 0 else 0.0


def _box_center(box):
  xyxy = _bbox_xyxy(box)
  if not xyxy:
    return None
  x1, y1, x2, y2 = xyxy
  return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_distance_normalized(a, b):
  ca = _box_center(a)
  cb = _box_center(b)
  if not ca or not cb:
    return float("inf")
  return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def _assign_teeth_to_quadrants_local(teeth, quadrants):
  assignments = {}
  for tooth_id, tooth in teeth.items():
    best_q = None
    best_iou = -1.0
    for q_name, q in quadrants.items():
      iou = _iou_xyxy(tooth.get("bbox"), q.get("bbox"))
      if iou > best_iou:
        best_iou = iou
        best_q = q_name
    if best_q is not None:
      assignments[tooth_id] = {
          "quadrant": best_q,
          "iou": round(best_iou, 6),
      }
  return assignments


def _human_tooth_name(quadrant, tooth_type):
  q = QUADRANT_DISPLAY.get(quadrant, quadrant or "")
  return f"{q} {tooth_type}".strip()


def _normalize_opgagent_teeth(yolo_json, quadrant_json):
  raw_teeth = yolo_json.get("detections", [])
  raw_quadrants = quadrant_json.get("detections", [])

  quadrants = {
      q.get("category_name"): {
          "name": q.get("category_name"),
          "bbox": q.get("bbox"),
          "confidence": q.get("confidence"),
      }
      for q in raw_quadrants
      if q.get("category_name")
  }

  teeth = {}
  for i, d in enumerate(raw_teeth, start=1):
    class_name = str(d.get("class_name"))
    teeth[str(i)] = {
        "tooth_type": TOOTH_TYPE_NAMES.get(class_name, f"class {class_name}"),
        "class_name": class_name,
        "confidence": d.get("confidence"),
        "bbox": d.get("bbox"),
    }

  assignments = _assign_teeth_to_quadrants_local(teeth, quadrants)

  normalized = []
  for tooth_id, tooth in teeth.items():
    a = assignments.get(tooth_id, {})
    quadrant = a.get("quadrant")
    normalized.append({
        "tooth_id": tooth_id,
        "tooth_type": tooth["tooth_type"],
        "quadrant": quadrant,
        "tooth_name": _human_tooth_name(quadrant, tooth["tooth_type"]),
        "confidence": tooth["confidence"],
        "bbox": tooth["bbox"],
        "quadrant_iou": a.get("iou"),
    })

  return {
      "image_name": yolo_json.get("image_name"),
      "image_size": yolo_json.get("image_size"),
      "tooth_count": len(normalized),
      "teeth": normalized,
      "quadrants": [
          {
              "name": q.get("category_name"),
              "confidence": q.get("confidence"),
              "bbox": q.get("bbox"),
          }
          for q in raw_quadrants
      ],
  }


def _associate_finding_to_tooth(finding, teeth):
  category = str(finding.get("category_name") or "").strip().lower()

  non_tooth_level = {
      "maxillary sinus",
      "mandibular canal",
      "bone loss",
      "missing teeth",
  }

  if category in non_tooth_level:
    return {
        "associated_tooth": None,
        "association_method": "not_tooth_level",
        "association_score": None,
        "association_iou": None,
        "association_center_distance_px": None,
    }

  fbox = finding.get("bbox")
  candidates = []

  for tooth in teeth:
    tbox = tooth.get("bbox")
    iou = _iou_xyxy(fbox, tbox)
    distance = _center_distance_normalized(fbox, tbox)
    candidates.append((iou, distance, tooth))

  if not candidates:
    return {
        "associated_tooth": None,
        "association_method": "no_tooth_candidates",
        "association_score": None,
        "association_iou": None,
        "association_center_distance_px": None,
    }

  best_iou, best_distance, best_tooth = max(
      candidates,
      key=lambda x: (x[0], -x[1]),
  )

  MIN_ASSOCIATION_IOU = 0.05
  MIN_LESION_CONTAINMENT = 0.50
  best_containment = 0.0

  lesion_box = finding.get("bbox") or finding.get("box")
  if lesion_box and best_tooth:
    lx1, ly1, lx2, ly2 = _bbox_xyxy(lesion_box)
    tx1, ty1, tx2, ty2 = _bbox_xyxy(best_tooth.get("bbox"))

    lesion_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    ix1 = max(lx1, tx1)
    iy1 = max(ly1, ty1)
    ix2 = min(lx2, tx2)
    iy2 = min(ly2, ty2)
    intersection_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    if lesion_area > 0:
      best_containment = intersection_area / lesion_area

  if (
      best_iou < MIN_ASSOCIATION_IOU
      and best_containment < MIN_LESION_CONTAINMENT
  ):
    return {
        "associated_tooth": None,
        "association_method": "insufficient_spatial_overlap",
        "association_score": round(best_iou, 6),
        "association_iou": round(best_iou, 6),
        "lesion_containment": round(best_containment, 6),
        "association_center_distance_px": round(best_distance, 2),
    }

  association_method = (
      "lesion_containment"
      if best_containment >= MIN_LESION_CONTAINMENT
      else "bbox_iou"
  )

  return {
      "associated_tooth": best_tooth.get("tooth_name"),
      "associated_tooth_id": best_tooth.get("tooth_id"),
      "associated_quadrant": best_tooth.get("quadrant"),
      "association_method": association_method,
      "association_score": round(best_iou, 6),
      "association_iou": round(best_iou, 6),
      "lesion_containment": round(best_containment, 6),
      "association_center_distance_px": round(best_distance, 2),
  }


def _normalize_tvem_findings(tvem_json, teeth):
  findings = []
  for d in tvem_json.get("detections", []):
    item = {
        "id": d.get("id"),
        "category_id": d.get("category_id"),
        "category_name": d.get("category_name"),
        "confidence": d.get("confidence"),
        "bbox": d.get("bbox"),
        "mask_area": d.get("mask_area"),
        "mask_shape": d.get("mask_shape"),
    }
    item.update(_associate_finding_to_tooth(item, teeth))
    findings.append(item)
  return findings


def run_opgagent(image_path):
  stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
  root_out = OPGAGENT_OUTPUT_DIR / stem
  root_out.mkdir(parents=True, exist_ok=True)

  yolo_out = root_out / "tooth_enumeration"
  yolo_json_path = yolo_out / f"{stem}_result.json"
  yolo_out.mkdir(parents=True, exist_ok=True)

  if not yolo_json_path.exists():
    _run_opgagent(
        [
            str(OPGAGENT_PYTHON),
            "predict.py",
            "--image",
            str(image_path),
            "--model",
            str(OPGAGENT_YOLO_MODEL),
            "--conf",
            "0.25",
            "--output",
            str(yolo_out),
        ],
        OPGAGENT_YOLO,
        "tooth enumeration",
    )
  else:
    print(f"[OPGAgent] Reusing tooth JSON: {yolo_json_path}")

  yolo_json = _read_json(yolo_json_path)

  tvem_jsons = {}
  for name, spec in OPGAGENT_TVEM_MODELS.items():
    out_dir = root_out / name
    json_path = out_dir / f"{stem}_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
      _run_opgagent(
          [
              str(OPGAGENT_PYTHON),
              "infer_with_json.py",
              "--config",
              str(spec["config"]),
              "--checkpoint",
              str(spec["checkpoint"]),
              "--image",
              str(image_path),
              "--output-dir",
              str(out_dir),
              "--category-file",
              str(spec["category_file"]),
              "--confidence",
              "0.30",
              "--device",
              "cuda",
          ],
          OPGAGENT_TVEM,
          name,
      )
    else:
      print(f"[OPGAgent] Reusing {name} JSON: {json_path}")

    tvem_jsons[name] = _read_json(json_path)

  tooth_info = _normalize_opgagent_teeth(
      yolo_json,
      tvem_jsons["4quadrants"],
  )

  associated = {
      name: _normalize_tvem_findings(data, tooth_info["teeth"])
      for name, data in tvem_jsons.items()
      if name != "4quadrants"
  }

  opgagent = {
      "pipeline": "OPGAgent",
      "image": str(image_path),
      "confidence_thresholds": {
          "tooth_enumeration": 0.25,
          "tvem": 0.30,
      },
      "tooth_enumeration": tooth_info,
      "findings": associated,
      "quadrants": tooth_info["quadrants"],
      "raw_json_paths": {
          "tooth_enumeration": str(yolo_json_path),
          **{
              name: str(root_out / name / f"{stem}_results.json")
              for name in tvem_jsons
          },
      },
  }

  relevant = []
  for name, items in associated.items():
    for item in items:
      relevant.append({
          "source": name,
          "finding": item["category_name"],
          "confidence": item["confidence"],
          "tooth": item.get("associated_tooth"),
          "quadrant": item.get("associated_quadrant"),
          "association_method": item.get("association_method"),
          "association_iou": item.get("association_iou"),
      })

  opgagent["clinically_relevant_summary"] = relevant
  return opgagent


def interactive_target_tooth_gate(opgagent_json, vision_json=None):
  tooth_data = opgagent_json.get("tooth_enumeration", {})
  teeth = tooth_data.get("teeth", [])
  all_findings = opgagent_json.get("clinically_relevant_summary", [])

  print("\n" + "=" * 90)
  print("STEP 2 — RADIOGRAPHIC FINDINGS REVIEW")
  print("=" * 90)

  print(
      "\nThe following findings were detected from the OPG by the"
      " radiographic AI models."
  )
  print(
      "These findings are observations only and have NOT yet been identified as"
      " the cause of the patient's symptoms."
  )

  print("\n" + "-" * 90)
  print("FROZEN ENDOLLM RADIOGRAPHIC MODELS")
  print("-" * 90)

  if vision_json:
    findings = vision_json.get("findings", {})
    lesions = findings.get("lesions", [])

    if lesions:
      print(f"\nPeriapical lesion(s) detected: {len(lesions)}")
      for i, lesion in enumerate(lesions, 1):
        bbox = lesion.get("bbox")
        localization = lesion.get("localization_confidence")
        radiographic_finding = lesion.get("radiographic_finding", {})
        pai_class = radiographic_finding.get("pai_class")

        print(f"\n  Lesion {i}:")
        if bbox is not None:
          print(f"    • Bounding box: {bbox}")
        if localization is not None:
          print(f"    • Localization confidence: {float(localization):.3f}")
        if pai_class is not None:
          print(f"    • PAI class: {pai_class}")

        pai_probabilities = radiographic_finding.get("pai_probabilities")
        if pai_probabilities:
          print("    • PAI probabilities:")
          for key, value in pai_probabilities.items():
            try:
              print(f"        {key}: {float(value):.3f}")
            except (TypeError, ValueError):
              print(f"        {key}: {value}")
    else:
      print("\nNo periapical lesion detected by the frozen lesion model.")
  else:
    print("\nFrozen-model output was not supplied to the target gate.")

  print("\n" + "-" * 90)
  print("OPGAGENT TOOTH ENUMERATION")
  print("-" * 90)

  if teeth:
    print(f"\nTotal teeth detected: {len(teeth)}")
    for tooth in teeth:
      print(
          f"  • {tooth.get('tooth_name', 'Unknown tooth')} (confidence"
          f" {float(tooth.get('confidence', 0.0)):.3f})"
      )
  else:
    print("\nNo teeth were detected by the OPGAgent tooth model.")

  print("\n" + "-" * 90)
  print("OPGAGENT ADDITIONAL RADIOGRAPHIC FINDINGS")
  print("-" * 90)

  if all_findings:
    print(f"\nTotal normalized OPGAgent findings: {len(all_findings)}")
    for finding in all_findings:
      name = finding.get("finding", "Unknown finding")
      confidence = finding.get("confidence")
      line = f"  • {name}"
      if confidence is not None:
        line += f" (confidence {float(confidence):.3f})"

      associated_tooth = finding.get("tooth")
      quadrant = finding.get("quadrant")
      if associated_tooth:
        line += f" — associated tooth: {associated_tooth}"
      elif quadrant:
        line += f" — region: {quadrant}"

      method = finding.get("association_method")
      iou = finding.get("association_iou")
      if method:
        line += f" [{method}"
        if iou is not None:
          line += f", IoU {float(iou):.3f}"
        line += "]"
      print(line)
  else:
    print("\nNo additional OPGAgent findings were detected.")

  print("\n" + "-" * 90)
  print(
      "IMPORTANT: The presence of multiple radiographic findings does not mean"
      " that all findings are related to the patient's symptoms."
  )

  if not teeth:
    print(
        "\nNo tooth-level target can be established from the OPG. Specialist"
        " intervention is required."
    )
    return {
        "status": "NO_TEETH_DETECTED",
        "confirmed": False,
        "target_teeth": [],
        "targeted_findings": [],
        "specialist_intervention_required": True,
    }

  tooth_lookup = {}
  for tooth in teeth:
    name = str(tooth.get("tooth_name", "")).strip()
    if name:
      tooth_lookup[name.lower()] = tooth

  print("\n" + "=" * 90)
  print("PATIENT-SPECIFIC TARGET TOOTH IDENTIFICATION")
  print("=" * 90)
  print(
      "\nWhich tooth/teeth are currently causing the patient's symptoms,"
      " discomfort, or pain?"
  )
  print("Please enter the human-readable tooth name(s), separated by commas.")
  print("Example: Lower left first molar")
  print("Type END to cancel.\n")

  while True:
    raw_target = input("TARGET TOOTH(S): ").strip()
    if raw_target.upper() == "END":
      return {
          "status": "CANCELLED",
          "confirmed": False,
          "target_teeth": [],
          "targeted_findings": [],
          "specialist_intervention_required": True,
      }

    if not raw_target:
      print("Please enter at least one tooth.")
      continue

    requested_names = [
        x.strip()
        for x in re.split(r",|;|\band\b", raw_target, flags=re.IGNORECASE)
        if x.strip()
    ]

    selected_teeth = []
    failed = False

    for requested in requested_names:
      key = requested.lower()
      if key in tooth_lookup:
        selected_teeth.append(tooth_lookup[key])
        continue

      matches = [
          tooth
          for name, tooth in tooth_lookup.items()
          if key in name or name in key
      ]

      if len(matches) == 1:
        selected_teeth.append(matches[0])
      elif len(matches) > 1:
        print(
            f"\nAmbiguous tooth description: '{requested}'. Please use the"
            " exact human-readable tooth name."
        )
        failed = True
        break
      else:
        print(
            f"\nThe requested tooth '{requested}' was not identified by the"
            " tooth-detection model."
        )
        failed = True
        break

    if not failed and selected_teeth:
      break

    print(
        "\nPlease enter the target tooth using one of the detected"
        " human-readable tooth names."
    )

  unique_teeth = []
  seen_ids = set()
  for tooth in selected_teeth:
    tooth_id = str(tooth.get("tooth_id", ""))
    if tooth_id not in seen_ids:
      unique_teeth.append(tooth)
      seen_ids.add(tooth_id)

  selected_names = {
      str(tooth.get("tooth_name", "")).strip().lower() for tooth in unique_teeth
  }

  targeted_findings = []
  for finding in all_findings:
    associated_tooth = str(finding.get("tooth") or "").strip().lower()
    if associated_tooth and associated_tooth in selected_names:
      targeted_findings.append(finding)

  frozen_target_findings = []

  def _bbox_iou_xyxy(box_a, box_b):
    try:
      ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
      bx1, by1, bx2, by2 = [float(v) for v in box_b]
    except (TypeError, ValueError):
      return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
      return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

  if vision_json:
    lesions = vision_json.get("findings", {}).get("lesions", [])
    for lesion in lesions:
      lesion_bbox = (
          lesion.get("bbox")
          or lesion.get("box")
          or lesion.get("bounding_box")
          or lesion.get("location", {}).get("bbox")
      )
      if not lesion_bbox:
        continue

      best_tooth = None
      best_iou = 0.0

      for tooth in unique_teeth:
        tooth_bbox = tooth.get("bbox")
        if not tooth_bbox:
          continue

        iou = _bbox_iou_xyxy(
            lesion_bbox,
            [
                tooth_bbox.get("x1"),
                tooth_bbox.get("y1"),
                tooth_bbox.get("x2"),
                tooth_bbox.get("y2"),
            ],
        )

        if iou > best_iou:
          best_iou = iou
          best_tooth = tooth

      if best_tooth is not None and best_iou > 0:
        lesion_copy = dict(lesion)
        lesion_copy["target_tooth"] = best_tooth.get("tooth_name")
        lesion_copy["target_tooth_id"] = best_tooth.get("tooth_id")
        lesion_copy["target_quadrant"] = best_tooth.get("quadrant")
        lesion_copy["association_method"] = "frozen_lesion_tooth_bbox_iou"
        lesion_copy["association_iou"] = best_iou
        frozen_target_findings.append(lesion_copy)

  print("\n" + "=" * 90)
  print("TARGET-SPECIFIC RADIOGRAPHIC REVIEW")
  print("=" * 90)
  print("\nUser-identified tooth/teeth:")
  for tooth in unique_teeth:
    print(
        f"  • {tooth.get('tooth_name', 'Unknown tooth')} (confidence"
        f" {float(tooth.get('confidence', 0.0)):.3f})"
    )

  if targeted_findings:
    print("\nOPGAgent findings associated with the selected tooth/teeth:")
    for finding in targeted_findings:
      confidence = float(finding.get("confidence", 0.0))
      line = (
          f"  • {finding.get('finding', 'Unknown finding')} (confidence"
          f" {confidence:.3f})"
      )
      method = finding.get("association_method")
      iou = finding.get("association_iou")
      if method:
        line += f" [{method}"
        if iou is not None:
          line += f", IoU {float(iou):.3f}"
        line += "]"
      print(line)
  else:
    print(
        "\nNo OPGAgent tooth-associated findings were detected for the selected"
        " tooth/teeth."
    )

  if frozen_target_findings:
    print(
        "\nFrozen-model periapical lesion finding(s) requiring target-tooth"
        " spatial reconciliation:"
    )
    for i, lesion in enumerate(frozen_target_findings, 1):
      localization = lesion.get("localization_confidence")
      pai = lesion.get("radiographic_finding", {}).get("pai_class")
      line = f"  • Periapical lesion {i}"
      if pai is not None:
        line += f" — PAI {pai}"
      if localization is not None:
        line += f" — localization confidence {float(localization):.3f}"
      print(line)

  if not targeted_findings and not frozen_target_findings:
    print(
        "\nNO RELEVANT OPG FINDINGS have currently been identified for the"
        " selected tooth/teeth."
    )
    print(
        "Specialist clinical intervention is required before an endodontic"
        " diagnosis can be generated."
    )

  print("\n" + "-" * 90)
  print(
      "Are these the tooth/teeth and radiographic findings relevant to the"
      " current complaint?"
  )
  print("Type YES or NO.\n")

  while True:
    confirmation = input("CONFIRMATION: ").strip().upper()
    if confirmation in {"YES", "Y"}:
      return {
          "status": "CONFIRMED",
          "confirmed": True,
          "target_teeth": unique_teeth,
          "targeted_findings": targeted_findings,
          "frozen_target_findings": frozen_target_findings,
          "specialist_intervention_required": (
              not targeted_findings and not frozen_target_findings
          ),
      }
    if confirmation in {"NO", "N"}:
      print(
          "\nPlease identify the correct tooth/teeth. The system will repeat"
          " the target-specific review."
      )
      return {
          "status": "RESELECT_REQUIRED",
          "confirmed": False,
          "target_teeth": unique_teeth,
          "targeted_findings": targeted_findings,
          "frozen_target_findings": frozen_target_findings,
          "specialist_intervention_required": False,
      }
    print("Please type YES or NO.")


def build_target_specific_evidence(
    vision_json,
    opgagent_json,
    target_context,
):
  target_context = target_context or {}
  target_teeth = target_context.get("target_teeth", [])
  targeted_opgagent_findings = target_context.get("targeted_findings", [])
  frozen_target_findings = target_context.get("frozen_target_findings", [])

  full_lesions = vision_json.get("findings", {}).get("lesions", [])
  target_lesion_ids = {
      str(x.get("lesion_id"))
      for x in frozen_target_findings
      if x.get("lesion_id") is not None
  }

  target_lesions = [
      dict(lesion)
      for lesion in full_lesions
      if lesion.get("lesion_id") is not None
      and str(lesion.get("lesion_id")) in target_lesion_ids
  ]

  targeted_vision = dict(vision_json)
  targeted_findings = dict(vision_json.get("findings", {}))
  targeted_findings["lesions"] = target_lesions
  targeted_findings["lesion_count"] = len(target_lesions)

  targeted_vision["findings"] = targeted_findings
  targeted_vision["target_scope"] = {
      "target_confirmed": True,
      "target_teeth": target_teeth,
      "target_lesion_ids": sorted(target_lesion_ids),
      "scope_rule": (
          "Only radiographic lesions spatially associated with the"
          " user-selected symptomatic tooth/teeth are available for downstream"
          " diagnostic reasoning."
      ),
  }

  non_tooth_level_findings = [
      dict(item)
      for item in opgagent_json.get("clinically_relevant_summary", [])
      if not item.get("tooth")
  ]

  target_opgagent = dict(opgagent_json)
  target_opgagent["clinically_relevant_summary"] = [
      dict(x) for x in targeted_opgagent_findings
  ]
  target_opgagent["target_context"] = {
      "target_teeth": target_teeth,
      "targeted_findings": targeted_opgagent_findings,
      "frozen_target_findings": frozen_target_findings,
      "non_tooth_level_context": non_tooth_level_findings,
      "scope_rule": (
          "Only tooth-level OPGAgent findings associated with the user-selected"
          " tooth/teeth are supplied as target evidence."
      ),
  }

  return {
      "target_teeth": target_teeth,
      "vision": targeted_vision,
      "opgagent": target_opgagent,
      "targeted_opgagent_findings": targeted_opgagent_findings,
      "frozen_target_findings": frozen_target_findings,
      "non_tooth_level_context": non_tooth_level_findings,
  }


def build_combined_visual_evidence(vision_json, opgagent_json):
  return {
      "existing_v62_vision": vision_json,
      "opgagent": opgagent_json,
      "scope": (
          "Periapical-lesion focused. OPGAgent findings are additive"
          " radiographic evidence."
      ),
  }


def assess_periapical_discordance(
    vision_json,
    opgagent_json,
    clinical_structured,
):
  lesions = vision_json.get("findings", {}).get("lesions", [])
  flags = []

  if not lesions:
    return {"flagged": False, "level": "none", "flags": []}

  tooth = clinical_structured.get("tooth_identified")
  cold = clinical_structured.get("cold_response")
  ept = clinical_structured.get("ept_response")
  isolated_deep_pocket = clinical_structured.get("isolated_deep_pocket")
  crack = clinical_structured.get("crack_suspected")
  vrf = clinical_structured.get("vertical_root_fracture_suspected")

  if tooth and cold is True:
    flags.append(
        "Radiographic periapical lesion is associated with a clinically"
        " responsive tooth on cold sensibility testing."
    )
  if tooth and ept is True:
    flags.append(
        "Radiographic periapical lesion is associated with a clinically"
        " responsive tooth on EPT."
    )
  if isolated_deep_pocket is True:
    flags.append(
        "An isolated deep periodontal pocket is present and may represent a"
        " competing periodontal or fracture-related finding."
    )
  if crack is True:
    flags.append(
        "Clinical findings raise concern for a cracked/fractured tooth."
    )
  if vrf is True:
    flags.append("Clinical findings raise concern for vertical root fracture.")

  if flags:
    return {
        "flagged": True,
        "level": (
            "high"
            if (cold is True and (vrf is True or crack is True))
            else "moderate"
        ),
        "flags": flags,
    }

  return {"flagged": False, "level": "none", "flags": []}


def classify_lesion_origin(
    clinical_structured,
    vision_json,
):
  """
  Stage 7.2 — Priority-ordered deterministic clinical origin gate.

  Gate priority:
    1. VRF
    2. Cracked tooth
    3. Standard endodontic origin
    4. Non-endodontic red flag
    5. Indeterminate

  Three-state rule:
    True  = explicitly positive
    False = explicitly negative
    None  = unknown / not established

  None is NEVER treated as False.
  """

  clinical_structured = clinical_structured or {}
  vision_json = vision_json or {}

  findings = vision_json.get("findings", {}) or {}
  lesions = findings.get("lesions", []) or []

  periapical_lesion_present = len(lesions) > 0

  previously_treated = (
      clinical_structured.get("previously_treated") is True
  )

  isolated_narrow_deep_pocket = (
      clinical_structured.get("isolated_narrow_deep_pocket") is True
  )

  sinus_tract_location = clinical_structured.get("sinus_tract_location")

  lesion_morphology = clinical_structured.get("lesion_morphology")

  normalized_morphology = (
      str(lesion_morphology).strip().lower()
      if lesion_morphology is not None
      else None
  )

  j_or_halo_morphology = normalized_morphology in {
      "j-shaped",
      "j shaped",
      "j-shape",
      "j shape",
      "halo",
      "j-shaped/halo",
      "j shaped/halo",
  }

  # ----------------------------------------------------------
  # GATE 1 — VRF SCREEN
  # ----------------------------------------------------------
  vrf_pocket_positive = isolated_narrow_deep_pocket
  vrf_sinus_positive = sinus_tract_location == "coronal/lateral"
  vrf_morphology_positive = j_or_halo_morphology

  vrf_triggered = (
      previously_treated
      and (
          vrf_pocket_positive
          or vrf_sinus_positive
          or vrf_morphology_positive
      )
  )

  vrf_basis = []

  if previously_treated:
    vrf_basis.append("previously_treated")

  if vrf_pocket_positive:
    vrf_basis.append("isolated_narrow_deep_pocket")

  if vrf_sinus_positive:
    vrf_basis.append("sinus_tract_location_coronal_lateral")

  if vrf_morphology_positive:
    vrf_basis.append("lesion_morphology_J_shaped_or_halo")

  if vrf_triggered:
    return {
        "category": "VRF_SUSPECTED",
        "priority": 1,
        "gate": "GATE_1_VRF",
        "confidence": "indeterminate",
        "basis": vrf_basis,
        "periapical_lesion_present": periapical_lesion_present,
        "gate_results": {
            "gate_1_vrf": True,
            "gate_2_cracked_tooth": False,
            "gate_3_endodontic_origin": False,
            "gate_4_non_endodontic_red_flag": False,
        },
        "clinical_requirements": [
            "Clinical periodontal probing and structural assessment.",
            "Consider CBCT when clinically indicated.",
            "Do not assume that endodontic retreatment will resolve a suspected VRF.",
        ],
        "reason": (
            "A previously treated tooth has a VRF-associated clinical or "
            "radiographic feature. Vertical root fracture must be screened "
            "before assigning a cracked-tooth or standard endodontic origin."
        ),
    }

  # ----------------------------------------------------------
  # GATE 2 — CRACKED TOOTH SCREEN
  # ----------------------------------------------------------
  pain_on_release = clinical_structured.get("pain_on_release") is True
  difficulty_localizing = (
      clinical_structured.get("difficulty_localizing_pain") is True
  )
  visible_crack_line = (
      clinical_structured.get("visible_crack_line") is True
  )

  bite_test_release_pain = pain_on_release

  crack_release_pattern = (
      bite_test_release_pain
      and difficulty_localizing
  )

  crack_triggered = (
      crack_release_pattern
      or visible_crack_line
  )

  crack_basis = []

  if crack_release_pattern:
    crack_basis.extend([
        "pain_on_release",
        "difficulty_localizing_pain",
    ])

  if visible_crack_line:
    crack_basis.append("visible_crack_line")

  if crack_triggered:
    return {
        "category": "CRACKED_TOOTH_SUSPECTED",
        "priority": 2,
        "gate": "GATE_2_CRACKED_TOOTH",
        "confidence": "indeterminate",
        "basis": crack_basis,
        "periapical_lesion_present": periapical_lesion_present,
        "gate_results": {
            "gate_1_vrf": False,
            "gate_2_cracked_tooth": True,
            "gate_3_endodontic_origin": False,
            "gate_4_non_endodontic_red_flag": False,
        },
        "clinical_requirements": [
            "Perform targeted crack assessment.",
            "Correlate bite/release findings with clinical examination.",
            "Assess periodontal probing and crack-line visibility."
        ],
        "reason": (
            "Clinical findings meet the Stage 7.2 cracked-tooth screening "
            "criteria. The finding remains a clinical suspicion requiring "
            "structural assessment."
        ),
    }

  # ----------------------------------------------------------
  # GATE 3 — STANDARD ENDODONTIC ORIGIN
  # ----------------------------------------------------------
  cold_negative = clinical_structured.get("cold_response") is False
  ept_negative = clinical_structured.get("ept_response") is False

  endodontic_origin_supported = (
      cold_negative
      and ept_negative
  )

  if endodontic_origin_supported:
    return {
        "category": "ENDODONTIC_ORIGIN_SUPPORTED",
        "priority": 3,
        "gate": "GATE_3_ENDODONTIC_ORIGIN",
        "confidence": "supported",
        "basis": [
            "cold_response_negative",
            "ept_response_negative",
        ],
        "periapical_lesion_present": periapical_lesion_present,
        "gate_results": {
            "gate_1_vrf": False,
            "gate_2_cracked_tooth": False,
            "gate_3_endodontic_origin": True,
            "gate_4_non_endodontic_red_flag": False,
        },
        "clinical_requirements": [],
        "reason": (
            "Both recorded sensibility tests are explicitly negative and "
            "no higher-priority VRF or cracked-tooth gate was triggered."
        ),
    }

  # ----------------------------------------------------------
  # GATE 4 — NON-ENDODONTIC RED FLAG
  # ----------------------------------------------------------
  cold_positive = clinical_structured.get("cold_response") is True
  ept_positive = clinical_structured.get("ept_response") is True

  non_endodontic_vitality_pattern = (
      cold_positive
      or ept_positive
  )

  non_endodontic_triggered = (
      non_endodontic_vitality_pattern
      and periapical_lesion_present
  )

  if non_endodontic_triggered:
    basis = []

    if cold_positive:
      basis.append("cold_response_positive")

    if ept_positive:
      basis.append("ept_response_positive")

    basis.append("periapical_lesion_present")

    return {
        "category": "NON_ENDODONTIC_RED_FLAG",
        "priority": 4,
        "gate": "GATE_4_NON_ENDODONTIC_RED_FLAG",
        "confidence": "indeterminate",
        "basis": basis,
        "periapical_lesion_present": periapical_lesion_present,
        "gate_results": {
            "gate_1_vrf": False,
            "gate_2_cracked_tooth": False,
            "gate_3_endodontic_origin": False,
            "gate_4_non_endodontic_red_flag": True,
        },
        "clinical_requirements": [
            "Confirm sensibility findings clinically.",
            "Evaluate the clinical-radiographic discordance.",
            "Consider non-endodontic or alternative pathology when appropriate."
        ],
        "reason": (
            "The tooth demonstrates a positive sensibility response despite "
            "a target-associated periapical lesion, without activation of the "
            "higher-priority VRF or cracked-tooth gates."
        ),
    }

  # ----------------------------------------------------------
  # FALLBACK
  # ----------------------------------------------------------
  return {
      "category": "INDETERMINATE",
      "priority": 5,
      "gate": "FALLBACK",
      "confidence": "indeterminate",
      "basis": [],
      "periapical_lesion_present": periapical_lesion_present,
      "gate_results": {
          "gate_1_vrf": False,
          "gate_2_cracked_tooth": False,
          "gate_3_endodontic_origin": False,
          "gate_4_non_endodontic_red_flag": False,
      },
      "clinical_requirements": [
          "Additional clinical diagnostic information is required."
      ],
      "reason": (
          "None of the ordered Stage 7.2 clinical origin gates was "
          "sufficiently established."
      ),
  }


def merge_red_flag_assessment(
    red_flag_result,
    discordance,
    clinical_structured,
    origin_classification=None,
):
  """
  Stage 7.2 integration layer.

  The unified classify_lesion_origin() function is the authoritative
  ordered clinical-origin gate. Existing red-flag assessment and
  escalation information are preserved as supporting layers, but they
  no longer independently override the Stage 7.2 priority order.
  """

  result = dict(red_flag_result)
  result["periapical_discordance"] = discordance

  clinical_structured = clinical_structured or {}

  if origin_classification is None:
    origin_classification = classify_lesion_origin(
        clinical_structured=clinical_structured,
        vision_json={},
    )

  result["lesion_origin_classification"] = origin_classification

  triage = dict(result.get("triage_decision", {}))

  category = origin_classification.get("category", "INDETERMINATE")

  # Preserve the prior category for auditability.
  triage["category_before_stage7_2"] = triage.get("category")

  # Stage 7.2 category is authoritative.
  triage["category"] = category
  triage["priority"] = origin_classification.get("priority")
  triage["gate"] = origin_classification.get("gate")
  triage["confidence"] = origin_classification.get(
      "confidence",
      "indeterminate",
  )
  triage["basis"] = list(origin_classification.get("basis", []))
  triage["reason"] = origin_classification.get("reason")

  triage["stage7_2_gate_results"] = origin_classification.get(
      "gate_results",
      {},
  )

  # Biological discordance remains separately visible for audit.
  triage["periapical_discordance_flag"] = bool(
      discordance.get("flagged")
  )

  # The Stage 7.2 category controls the red-flag flag.
  triage["non_endodontic_cause_must_be_considered"] = (
      category == "NON_ENDODONTIC_RED_FLAG"
  )

  # Structural/crack/VRF cases are not relabeled as non-endodontic merely
  # because a radiographic lesion exists.
  if category in {
      "VRF_SUSPECTED",
      "CRACKED_TOOTH_SUSPECTED",
  }:
    triage["non_endodontic_cause_must_be_considered"] = False

  result["triage_decision"] = triage
  return result



def normalize_clinical_language(text: str) -> str:
  """Normalizes common clinician terminology, typos, and syntax to canonical phrases

  prior to semantic parsing.
  Negation prefixes are strictly preserved.
  """
  if not text:
    return ""

  h = str(text)

  # Canonical substitutions mapping typos/synonyms
  patterns: List[Tuple[str, str]] = [
      # Typos & phonetic spelling
      (r"\bbitting\b", "biting"),
      (r"\bbiten\b", "biting"),
      (r"\bchewwn\b", "chewing"),
      (r"\bchewd\b", "chewing"),
      # Mastication & pressure synonyms
      (r"\b(?:on|upon|while|during)\s+mastication\b", "on chewing"),
      (r"\bmasticatory\s+pain\b", "pain on chewing"),
      (r"\bocclusal\s+(?:load|loading|pressure)\b", "biting pressure"),
      (r"\bocclusal\s+tenderness\b", "pain on biting"),
      (r"\btender\s+to\s+(?:bite|chew|chewing)\b", "pain on biting"),
      (r"\bpainful\s+(?:to|on)\s+(?:bite|chew)\b", "pain on biting"),
      (r"\bdiscomfort\s+(?:on|during|when)\s+biting\b", "pain on biting"),
      (r"\bdiscomfort\s+(?:on|during|when)\s+chewing\b", "pain on chewing"),
      (r"\bpain\s+with\s+chewing\b", "pain on chewing"),
      # Rebound / release phenomena
      (r"\brebound\s+pain\b", "pain on release"),
      (
          r"\b(?:pain|discomfort)\s+(?:upon|on|when|after)\s+releas(?:e|ing)\b",
          "pain on release",
      ),
      (
          r"\bdisappears?\s+(?:upon|on|when|after)\s+releas(?:e|ing)\b",
          "pain on release",
      ),
      (
          r"\bceases?\s+(?:upon|on|when|after)\s+releas(?:e|ing)\b",
          "pain on release",
      ),
      # Percussion / Palpation
      (r"\bpercussion\s+tender\b", "pain on percussion"),
      (r"\btender\s+to\s+percussion\b", "pain on percussion"),
      (r"\bpalpation\s+tender\b", "pain on palpation"),
      (r"\btender\s+to\s+palpation\b", "pain on palpation"),
      # Cold testing / lingering
      (r"\bcold\s+sensitivity\b", "cold response"),
      (r"\bresponse\s+to\s+cold\b", "cold response"),
      (r"\bthermal\s+test(?:ing)?\b", "cold response"),
      (
          r"\b(?:prolonged|persistent|lingering)\s+(?:pain|response)\s+(?:to|after)\s+cold\b",
          "cold lingering",
      ),
      (
          r"\bpain\s+(?:lingers|persists|continues)\s+after\s+cold\b",
          "cold lingering",
      ),
      # Electric pulp test (EPT)
      (
          r"\belectric(?:al)?\s+pulp\s+test(?:ing)?\b",
          "ept response",
      ),
      # Swelling & drainage
      (r"\b(?:facial|intraoral|gingival)\s+edema\b", "swelling"),
      (r"\bfluctuant\s+swelling\b", "swelling"),
      (r"\bgum\s+boil\b", "sinus tract"),
      (r"\bparulis\b", "sinus tract"),
      (r"\bdraining\s+(?:sinus|fistula)\b", "sinus tract"),
      # Periodontal pocketing
      # Preserve the narrower VRF-relevant pattern before the generic
      # isolated-deep-pocket normalization.
      # VRF-specific pattern requires explicit isolation.
      # Do NOT convert a generic "narrow deep pocket" into an isolated
      # narrow deep pocket because isolation is diagnostically important.
      (
          r"\bisolated\s+narrow\s+(?:deep\s+)?(?:periodontal\s+)?pocket\b",
          "isolated narrow deep pocket",
      ),
      (
          r"\bisolated\s+deep\s+(?:periodontal\s+)?pocket\b",
          "isolated deep pocket",
      ),
      (
          r"\bdeep\s+(?:periodontal\s+)?pocket[^.\n]*isolated\b",
          "isolated deep pocket",
      ),
      (
          r"\bpocket(?:ing)?\s+(?:depth\s+)?(?:>=|greater than|>)?\s*[6-9]\s*mm\b",
          "isolated deep pocket",
      ),
      # Pain localization
      (
          r"\b(?:poorly|badly|not well|unable to|cannot|can't)\s+"
          r"(?:locali[sz]e|locali[sz]ing|locali[sz]ed)\s+"
          r"(?:the\s+)?pain\b",
          "difficulty localizing pain",
      ),
      (
          r"\bpain\s+(?:is\s+)?(?:poorly|badly|not well)\s+locali[sz]ed\b",
          "difficulty localizing pain",
      ),
      (
          r"\bdifficult(?:y)?\s+(?:in\s+)?locali[sz](?:e|ing)\s+(?:the\s+)?pain\b",
          "difficulty localizing pain",
      ),

      # Visible crack line
      (
          r"\b(?:visible|directly visible|seen)\s+crack(?:\s+line)?\b",
          "visible crack line",
      ),

      # Fractures & Cracks
      (
          r"\b(?:suspected\s+)?(?:cracked\s+tooth|tooth\s+crack)\b",
          "crack suspected",
      ),
      (r"\bvertical\s+root\s+fracture\b", "vertical root fracture suspected"),
      (r"\bvrf\b", "vertical root fracture suspected"),
  ]

  for pat, rep in patterns:
    h = re.sub(pat, rep, h, flags=re.IGNORECASE)

  return h


def extract_clinical_structured(history_text: str) -> Dict[str, Any]:
  """Multi-stage clinical extractor with contextual clause segmentation and negation filtering."""
  raw_text = history_text.strip()
  normalized = normalize_clinical_language(raw_text)

  structured: Dict[str, Any] = {
      "tooth_identified": None,
      "chief_complaint": None,
      "symptom_onset_duration": None,
      "pain_on_biting": None,
      "pain_on_release": None,
      "difficulty_localizing_pain": None,
      "cold_response": None,
      "cold_lingering": None,
      "ept_response": None,
      "percussion": None,
      "palpation": None,
      "spontaneous_pain": None,
      "swelling": None,
      "sinus_tract": None,
      "sinus_tract_location": None,
      "isolated_deep_pocket": None,
      "isolated_narrow_deep_pocket": None,
      "bleeding_on_probing": None,
      "mobility": None,
      "furcation_involvement": None,
      "crack_suspected": None,
      "visible_crack_line": None,
      "vertical_root_fracture_suspected": None,
      "previously_treated": None,
      "lesion_morphology": None,
      "trauma_history": None,
      "systemic_infection_signs": None,
  }

  # 1. Chief Complaint & Duration Extraction
  cc_m = re.search(
      r"(?:chief complaint|chief concern|c/o|presenting complaint)\s*[:\-]\s*([^\n;]+)",
      raw_text,
      flags=re.IGNORECASE,
  )
  if cc_m:
    structured["chief_complaint"] = cc_m.group(1).strip()

  dur_m = re.search(
      r"(?:onset|duration|since|symptom duration)\s*[:\-]?\s*([^\n;]+)",
      raw_text,
      flags=re.IGNORECASE,
  )
  if dur_m:
    structured["symptom_onset_duration"] = dur_m.group(1).strip()

  # 2. Tooth Identification
  tooth_name_pat = r"\b(upper|lower)\s+(right|left)\s+(central incisor|lateral incisor|canine|first premolar|second premolar|first molar|second molar|third molar|wisdom tooth)\b"
  t_m = re.search(tooth_name_pat, normalized, flags=re.IGNORECASE)
  if t_m:
    arch = t_m.group(1).title()
    side = t_m.group(2).title()
    ttype = t_m.group(3).lower()
    if ttype == "wisdom tooth":
      ttype = "third molar"
    structured["tooth_identified"] = f"{arch} {side} {ttype}"
  else:
    fdi_m = re.search(r"\b(?:tooth\s*)?([1-4][0-8])\b", raw_text)
    if fdi_m:
      structured["tooth_identified"] = fdi_m.group(1)

  # 3. Contextual Clause Splitter (preserves negation scope)
  clauses = [
      c.strip().lower()
      for c in re.split(r"[\n,;]|\bbut\b|\band\b", normalized)
      if c.strip()
  ]

  # ---------------------------------------------------------------
  # STAGE 7.2 — J-SHAPED / HALO LESION MORPHOLOGY
  # ---------------------------------------------------------------
  # Explicit wording only. This is not inferred merely from the
  # presence of a periapical lesion.
  #
  # Examples recognized:
  #   J-shaped lesion
  #   J shaped lesion
  #   halo-shaped lesion
  #   halo appearance
  #   J-shaped or halo appearance
  #
  # The classifier later uses this field for the Gate 1 VRF screen.

  if re.search(
      r"\bj[-\s]?shaped\b"
      r"|\bhalo[-\s]?shaped\b"
      r"|\bhalo\s+(?:appearance|configuration|pattern)\b"
      r"|\bj[-\s]?shaped\s+(?:or|/)\s+halo\b",
      normalized.lower(),
  ):
      structured["lesion_morphology"] = "J-shaped/halo"

  def check_attribute(
      pos_patterns: List[str], neg_patterns: List[str]
  ) -> Optional[bool]:
    """
    Three-state clinical extraction:
      True  = explicit positive finding
      False = explicit negative finding
      None  = not established

    If the history contains genuinely conflicting positive and negative
    statements, return None rather than arbitrarily choosing one.
    """
    has_pos = False
    has_neg = False

    for c in clauses:
      # Detect explicit negative findings first.
      neg_matches = []
      for pattern in neg_patterns:
        for match in re.finditer(pattern, c):
          neg_matches.append(match.span())

      if neg_matches:
        has_neg = True

      # Mask text belonging to an explicit negative expression before
      # searching for positive expressions. This prevents a positive
      # phrase embedded inside a negative statement from creating a
      # false conflict.
      positive_search_text = c

      if neg_matches:
        chars = list(c)
        for start, end in neg_matches:
          for idx in range(start, end):
            chars[idx] = " "
        positive_search_text = "".join(chars)

      if any(re.search(pattern, positive_search_text)
             for pattern in pos_patterns):
        has_pos = True

    if has_pos and has_neg:
      return None
    if has_neg:
      return False
    if has_pos:
      return True
    return None

  # 4. Extract Structured Fields
  # Biting Pain
  structured["pain_on_biting"] = check_attribute(
      pos_patterns=[
          r"pain on biting",
          r"pain on chewing",
          r"pain while biting",
          r"pain when biting",
          r"pain while chewing",
          r"biting pressure",
          r"biting[^.\n]{0,30}pain",
      ],
      neg_patterns=[
          r"no pain on biting",
          r"no pain on chewing",
          r"painless on biting",
          r"not painful on biting",
          r"no biting pain",
      ],
  )

  # Release Pain (Cracked tooth signature)
  structured["pain_on_release"] = check_attribute(
      pos_patterns=[
          r"pain on release",
          r"pain upon release",
          r"pain after release",
          r"pain when (?:the )?patient releases? (?:the )?bite",
          r"pain\s+(?:is\s+)?(?:particularly\s+)?(?:severe|sharp|marked)\s+when\s+(?:the\s+)?patient\s+releases?\s+(?:the\s+)?bite",
          r"pain\s+(?:is\s+)?(?:particularly\s+)?(?:severe|sharp|marked)\s+when\s+(?:the\s+)?bite\s+is\s+released",
          r"pain when (?:the )?bite is released",
          r"pain when releasing (?:the )?bite",
          r"pain while releasing (?:the )?bite",
          r"pain on releasing (?:the )?bite",
          r"severe pain (?:when|upon|on) releasing",
          r"rebound pain",
          r"disappears upon releasing",
          r"pain.*releases? the bite",
      ],
      neg_patterns=[
          r"no pain on release",
          r"no rebound pain",
          r"no pain when (?:the )?patient releases? (?:the )?bite",
          r"no pain when (?:the )?bite is released",
          r"no pain on releasing (?:the )?bite",
      ],
  )

  # Pain Localization
  structured["difficulty_localizing_pain"] = check_attribute(
      pos_patterns=[
          r"difficulty localizing pain",
          r"poorly localized pain",
          r"badly localized pain",
          r"not well localized pain",
          r"unable to localize pain",
          r"cannot localize pain",
          r"can't localize pain",
          r"cannot clearly localize pain",
          r"cannot clearly identify (?:which )?tooth",
          r"can't clearly identify (?:which )?tooth",
          r"unable to identify (?:which )?tooth",
          r"unable to identify the (?:causative|causal|affected) tooth",
          r"cannot identify the (?:causative|causal|affected) tooth",
          r"cannot tell which tooth",
          r"can't tell which tooth",
          r"not able to identify which tooth",
          r"unable to tell which tooth",
          r"difficult to identify which tooth",
      ],
      neg_patterns=[
          r"pain is well localized",
          r"well localized pain",
          r"able to localize pain",
          r"localizes pain precisely",
          r"clearly identifies which tooth",
          r"able to identify which tooth",
          r"can identify which tooth",
      ],
  )

  # Isolated Deep Pocket
  structured["isolated_deep_pocket"] = check_attribute(
      pos_patterns=[
          r"isolated deep pocket",
          r"isolated periodontal pocket",
          r"single deep pocket",
          r"deep pocket",
      ],
      neg_patterns=[
          r"no isolated deep pocket",
          r"no deep pocket",
          r"probing depths normal",
          r"normal probing",
      ],
  )

  # Isolated Narrow Deep Pocket — VRF-specific probing pattern
  structured["isolated_narrow_deep_pocket"] = check_attribute(
      pos_patterns=[
          r"isolated narrow deep pocket",
          r"narrow deep pocket",
          r"isolated narrow pocket",
          r"narrow isolated pocket",
          r"isolated deep narrow pocket",
      ],
      neg_patterns=[
          r"no isolated narrow deep pocket",
          r"no narrow deep pocket",
          r"no isolated narrow pocket",
          r"probing depths are not narrow or isolated",
      ],
  )

  # Cold Response
  structured["cold_response"] = check_attribute(
      pos_patterns=[
          r"cold response is positive",
          r"positive cold response",
          r"cold response[^.\n]*positive",
          r"normal positive response",
          r"responds to cold",
          r"responds positively to cold",
          r"responded positively to cold",
          r"positive response to cold",
          r"positive response on cold testing",
          r"positive cold response",
      ],
      neg_patterns=[
          r"cold response is negative",
          r"no response to cold",
          r"does not respond to cold",
          r"did not respond to cold",
          r"does not respond to cold testing",
          r"did not respond to cold testing",
          r"no response on cold testing",
          r"negative cold",
          r"non-responsive to cold",
      ],
  )

  # Cold Lingering
  structured["cold_lingering"] = check_attribute(
      pos_patterns=[
          r"cold lingering",
          r"lingering pain after cold",
          r"prolonged cold",
      ],
      neg_patterns=[
          r"no lingering",
          r"does not linger",
          r"transient cold response",
      ],
  )

  # EPT
  structured["ept_response"] = check_attribute(
      pos_patterns=[
          r"ept response positive",
          r"positive ept",
          r"responds to ept",
      ],
      neg_patterns=[
          r"ept response negative",
          r"no response to ept",
          r"does not respond to ept",
          r"did not respond to ept",
          r"no response on ept",
          r"negative ept response",
      ],
  )

  # Percussion
  structured["percussion"] = check_attribute(
      pos_patterns=[
          r"pain on percussion",
          r"positive percussion",
          r"tender to percussion",
      ],
      neg_patterns=[
          r"no pain on percussion",
          r"negative percussion",
          r"not tender to percussion",
      ],
  )

  # Palpation
  structured["palpation"] = check_attribute(
      pos_patterns=[
          r"pain on palpation",
          r"positive palpation",
          r"tender to palpation",
      ],
      neg_patterns=[
          r"no pain on palpation",
          r"negative palpation",
          r"not tender to palpation",
      ],
  )

  # Spontaneous Pain
  structured["spontaneous_pain"] = check_attribute(
      pos_patterns=[
          r"spontaneous pain",
          r"spontaneous toothache",
          r"night pain",
      ],
      neg_patterns=[
          r"no spontaneous pain",
          r"no spontaneous discomfort",
          r"no pain at rest",
      ],
  )

  # Swelling
  structured["swelling"] = check_attribute(
      pos_patterns=[
          r"swelling",
          r"intraoral swelling",
          r"facial swelling",
          r"vestibular swelling",
      ],
      neg_patterns=[r"no swelling", r"absence of swelling", r"without swelling"],
  )

  # Sinus Tract
  structured["sinus_tract"] = check_attribute(
      pos_patterns=[
          r"sinus tract",
          r"parulis",
          r"draining tract",
          r"fistula present",
      ],
      neg_patterns=[
          r"no sinus tract",
          r"sinus tract absent",
          r"without sinus tract",
      ],
  )

  # Sinus Tract Location
  # Location is only assigned when explicitly described. A sinus tract
  # without stated location remains "unknown"; it is never inferred.
  sinus_present = structured.get("sinus_tract") is True

  coronal_sinus_patterns = [
      r"sinus tract[^.\n]{0,80}\bcoronal\b",
      r"sinus tract[^.\n]{0,80}\blateral\b",
      r"\bcoronal\b[^.\n]{0,80}sinus tract",
      r"\blateral\b[^.\n]{0,80}sinus tract",
      r"\bdraining tract[^.\n]{0,80}\bcoronal\b",
      r"\bdraining tract[^.\n]{0,80}\blateral\b",
      r"\bsinus tract[^.\n]{0,80}\balong the lateral root\b",
      r"\bsinus tract[^.\n]{0,80}\bcoronal to the apex\b",
      r"\bsinus tract[^.\n]{0,80}\blateral to the apex\b",
  ]

  apical_sinus_patterns = [
      r"sinus tract[^.\n]{0,80}\bapical\b",
      r"sinus tract[^.\n]{0,80}\bat the apex\b",
      r"\bapical\b[^.\n]{0,80}sinus tract",
      r"\bapex\b[^.\n]{0,80}sinus tract",
  ]

  has_coronal = any(
      re.search(pat, normalized, flags=re.IGNORECASE)
      for pat in coronal_sinus_patterns
  )
  has_apical = any(
      re.search(pat, normalized, flags=re.IGNORECASE)
      for pat in apical_sinus_patterns
  )

  if has_coronal and has_apical:
    structured["sinus_tract_location"] = "indeterminate"
  elif has_coronal:
    structured["sinus_tract_location"] = "coronal/lateral"
  elif has_apical:
    structured["sinus_tract_location"] = "apical"
  elif sinus_present:
    structured["sinus_tract_location"] = "unknown"
  else:
    structured["sinus_tract_location"] = None

  # Crack / VRF
  structured["crack_suspected"] = check_attribute(
      pos_patterns=[r"crack suspected", r"cracked tooth"],
      neg_patterns=[r"no crack", r"no cracked tooth"],
  )

  structured["vertical_root_fracture_suspected"] = check_attribute(
      pos_patterns=[
          r"vertical root fracture suspected",
          r"root fracture suspected",
      ],
      neg_patterns=[r"no vertical root fracture", r"no vrf"],
  )

  structured["visible_crack_line"] = check_attribute(
      pos_patterns=[
          r"visible crack line",
          r"visible crack",
          r"directly visible crack",
          r"seen crack line",
      ],
      neg_patterns=[
          r"no visible crack",
          r"no visible crack line",
          r"crack line not visible",
          r"no crack line seen",
      ],
  )

  # Previous Treatment
  structured["previously_treated"] = check_attribute(
      pos_patterns=[
          r"previously treated",
          r"previously root canal treated",
          r"prior root canal",
          r"previous rct",
          r"endodontically treated",
      ],
      neg_patterns=[
          r"not previously treated",
          r"has not been previously treated",
          r"has not been previously root canal treated",
          r"has not been previously root canal treatment",
          r"has not been root canal treated",
          r"has not undergone root canal treatment",
          r"not previously root canal treated",
          r"not previously endodontically treated",
          r"no previous root canal",
          r"no prior rct",
          r"no previous rct",
          r"never previously treated",
          r"never root canal treated",
          r"has never been root canal treated",
          r"has not undergone previous root canal treatment",
          r"no history of root canal treatment",
      ],
  )

  # Trauma
  structured["trauma_history"] = check_attribute(
      pos_patterns=[r"trauma history", r"history of trauma", r"recent trauma"],
      neg_patterns=[r"no trauma history", r"no history of trauma"],
  )

  # Systemic Signs
  structured["systemic_infection_signs"] = check_attribute(
      pos_patterns=[
          r"fever",
          r"lymphadenopathy",
          r"malaise",
          r"systemic symptoms",
      ],
      neg_patterns=[r"no fever", r"no systemic symptoms", r"afebrile"],
  )

  # IMPORTANT:
  # Pain on release alone must NOT automatically establish crack suspicion.
  # Stage 7.2 determines CRACKED_TOOTH_SUSPECTED using the ordered gate:
  #   (pain on release AND difficulty localizing pain)
  #   OR visible crack line.
  # Therefore crack_suspected remains an explicitly documented finding only.

  return structured


def generate_history_questions(model, processor, visual_evidence):
  visual_text = json.dumps(visual_evidence, indent=2, ensure_ascii=False)

  prompt = f"""
You are an Endodontic clinician conducting a focused diagnostic history
for a patient with a radiographically detected periapical lesion.

The existing v62 radiographic system and an additive OPGAgent system have
already analyzed the image. Their structured visual evidence is below:

{visual_text}

Your ONLY task is to ask the clinician for the SMALLEST SET of
HIGH-VALUE CLINICAL INFORMATION that is NOT reliably established by the
visual evidence and that could materially change the probable pulpal,
apical, or etiologic interpretation.

The clinical scope is STRICTLY the differential diagnosis of a
radiographically detected periapical/periradicular lesion. Do not broaden
this into a general dental examination.

NEVER ask about detector confidence, PAI probabilities, bounding boxes, or AI models.

Prioritize, as relevant:
- which tooth/teeth clinically correspond to the lesion
- chief complaint and symptom onset/duration
- spontaneous versus provoked pain
- thermal/cold response and lingering
- EPT response
- comparison with adjacent/control teeth
- percussion and palpation
- biting pain, including pain on release and difficulty localizing the pain
- swelling, sinus tract or drainage, including whether a sinus tract opens
  apically or coronal/lateral to the root
- periodontal probing depths, especially an isolated narrow deep pocket
- previous root canal treatment and relevant post-treatment history
- clinical evidence of a cracked/fractured tooth, including any directly
  visible crack line

Maximum: 5 questions.
Do not provide a diagnosis.
Return ONLY a numbered list of clinician-facing questions.
"""
  return qwen_text(model, processor, prompt, max_new_tokens=300)


def collect_history(question_text):
  print("\n" + "=" * 90)
  print("STEP 2 — CLINICAL HISTORY")
  print("=" * 90)
  print("\nSuggested questions from the Endodontic reasoning model:\n")
  print(question_text)
  print("\n" + "-" * 90)
  print(
      "Enter the clinician's answers below. You may answer in one combined"
      " narrative."
  )
  print("Type END on a new line when finished.\n")

  lines = []
  while True:
    line = input()
    if line.strip().upper() == "END":
      break
    lines.append(line)

  return "\n".join(lines).strip()


# ============================================================
# RETRIEVAL QUERY
# ============================================================


def build_retrieval_query(
    vision_json,
    clinical_history,
    clinical_structured=None,
    non_endodontic=False,
    opgagent_json=None,
    target_context=None,
):
  lesions = vision_json.get("findings", {}).get("lesions", [])
  lesion_summary = []

  for lesion in lesions:
    pai = lesion.get("radiographic_finding", {}).get("pai_class")
    lesion_summary.append(f"Lesion {lesion.get('lesion_id')}: PAI {pai}")

  radiographic_summary = (
      "\n".join(lesion_summary) or "No lesion summary available."
  )
  target_context = target_context or {}
  opg = opgagent_json or {}

  target_teeth = target_context.get("target_teeth", [])
  target_names = [
      str(t.get("tooth_name", "")).strip()
      for t in target_teeth
      if t.get("tooth_name")
  ]

  opg_lines = []
  for item in opg.get("clinically_relevant_summary", []):
    line = (
        f"- {item.get('finding')} (confidence {item.get('confidence')},"
        f" associated tooth: {item.get('tooth')}, quadrant:"
        f" {item.get('quadrant')})"
    )
    opg_lines.append(line)

  frozen_target_findings = target_context.get("frozen_target_findings", [])
  frozen_lines = []
  for lesion in frozen_target_findings:
    rf = lesion.get("radiographic_finding", {})
    frozen_lines.append(
        f"- Frozen periapical lesion {lesion.get('lesion_id')}: PAI"
        f" {rf.get('pai_class')}, target tooth: {lesion.get('target_tooth')},"
        f" association IoU: {lesion.get('association_iou')}"
    )

  target_tooth_summary = (
      ", ".join(target_names) if target_names else "No target tooth recorded."
  )
  opg_summary = "\n".join(opg_lines) or "- No target-tooth OPGAgent finding."
  frozen_summary = (
      "\n".join(frozen_lines) or "- No target-associated frozen lesion."
  )

  # VERIFIED STRUCTURED CLINICAL PROFILE FOR DENSE RETRIEVAL
  clinical_structured = clinical_structured or {}

  excluded_fields = {
      "tooth_identified",
      "chief_complaint",
      "symptom_onset_duration",
  }

  struct_lines = []
  for key, value in clinical_structured.items():
    if value is not None and key not in excluded_fields:
      label = key.replace("_", " ").title()
      struct_lines.append(f"- {label}: {value}")

  structured_summary = (
      "\n".join(struct_lines)
      if struct_lines
      else "- No structured clinical findings recorded."
  )

  # Retrieval-expansion terms only; these do not constitute a diagnosis.
  thematic_queries = []

  if clinical_structured.get("previously_treated"):
    thematic_queries.append(
        "previously root canal treated tooth post-treatment apical disease "
        "persistent apical periodontitis endodontic treatment outcome"
    )

  if (
      clinical_structured.get("pain_on_release")
      or clinical_structured.get("crack_suspected")
  ):
    thematic_queries.append(
        "cracked tooth release pain pain on release biting release tenderness "
        "longitudinal fracture"
    )

  if (
      clinical_structured.get("isolated_deep_pocket")
      or clinical_structured.get("isolated_narrow_deep_pocket")
      or clinical_structured.get("vertical_root_fracture_suspected")
      or clinical_structured.get("sinus_tract_location") == "coronal/lateral"
  ):
    thematic_queries.append(
        "vertical root fracture isolated narrow deep periodontal pocket "
        "coronal lateral sinus tract J-shaped halo lesion previously treated tooth"
    )

  # Stage 7.2 hierarchy:
  # When the authoritative VRF screen is positive, pain on biting must not
  # create a competing symptomatic-apical-periodontitis retrieval branch.
  vrf_retrieval_priority = (
      clinical_structured.get("previously_treated") is True
      and (
          clinical_structured.get("isolated_narrow_deep_pocket") is True
          or clinical_structured.get("sinus_tract_location") == "coronal/lateral"
          or clinical_structured.get("lesion_morphology") == "J-shaped/halo"
      )
  )

  if (
      not vrf_retrieval_priority
      and (
          clinical_structured.get("pain_on_biting")
          or clinical_structured.get("percussion")
      )
  ):
    thematic_queries.append(
        "symptomatic apical periodontitis pain on biting chewing pressure "
        "percussion tenderness"
    )

  thematic_summary = " ".join(thematic_queries)

  base = f"""
Periapical lesion diagnostic reasoning.

TARGET TOOTH/TEETH:
{target_tooth_summary}

TARGET-ASSOCIATED FROZEN RADIOGRAPHIC FINDINGS:
{frozen_summary}

TARGET-ASSOCIATED OPGAGENT FINDINGS:
{opg_summary}
VERIFIED STRUCTURED CLINICAL PROFILE:

{structured_summary}

FOCUSED CLINICAL TOPICS FOR RETRIEVAL:

{thematic_summary}


IMPORTANT TARGET BOUNDARY:
Only findings explicitly associated with the user-selected tooth/teeth
are part of the diagnostic evidence. Unrelated tooth-level findings
from the remainder of the OPG must not influence retrieval or diagnosis.

Clinical history and examination:
{clinical_history}
""".strip()

  if non_endodontic:
    return f"""
{base}

Retrieve evidence concerning:
- endodontic versus non-endodontic origin of periapical/periradicular lesions
- tooth-lesion discordance
- positive sensibility/EPT response despite a periapical lesion
- periodontal and perio-endo presentations
- cracked tooth, fractured tooth, and vertical root fracture as
  competing causes of periapical/periodontal findings
- non-endodontic periapical mimics and red flags
""".strip()

  return f"""
{base}

Retrieve authoritative endodontic evidence relevant to:
- pulpal diagnosis and apical diagnosis
- symptomatic apical periodontitis vs asymptomatic apical periodontitis
- biting pressure, mastication pain, and release tenderness in apical and cracked-tooth diseases
- correlation of symptoms, sensibility/EPT, percussion/palpation, periodontal probing and radiographic findings
- AAE/ESE diagnostic guidance
""".strip()


def build_dynamic_clarifications(clinical_structured):
  notes = []

  if (
      clinical_structured.get("cold_response") is True
      and clinical_structured.get("spontaneous_pain") is False
  ):
    notes.append(
        '- The patient has no spontaneous pain or discomfort. A positive'
        ' response to the cold test is a positive pulpal sensibility response.'
        ' Do not convert this finding alone into a definitive pulpal diagnosis.'
        ' Frame your reasoning around the clinical-radiographic discordance and'
        ' preserve the programmatic pulpal diagnostic status.'
    )

  if (
      clinical_structured.get("crack_suspected") is True
      or clinical_structured.get("vertical_root_fracture_suspected") is True
      or clinical_structured.get("isolated_narrow_deep_pocket") is True
      or clinical_structured.get("sinus_tract_location") == "coronal/lateral"
  ):
    notes.append(
        '- Clinical findings raise structural/fracture-related concern.'
        ' The Stage 7.2 priority gate must be preserved and must not be'
        ' overridden by the presence of a periapical radiographic lesion.'
    )

  if clinical_structured.get("isolated_deep_pocket") is True:
    notes.append(
        '- An isolated deep periodontal pocket is present. Consider a'
        ' periodontal or fracture-related pathway alongside endodontic'
        ' disease.'
    )

  if (
      clinical_structured.get("pain_on_biting") is True
      or clinical_structured.get("pain_on_release") is True
  ):
    notes.append(
        '- Pain on biting, mastication pressure, and release tenderness are'
        ' canonical signs of periradicular inflammation (e.g. symptomatic'
        ' apical periodontitis) and/or structural tooth crack. Authoritative'
        ' retrieved literature explicitly validates this relationship; do NOT'
        ' characterize biting pain as atypical or inconsistent.'
    )

  return "\n".join(notes)


# ============================================================
# POST-GENERATION SAFETY VALIDATION
# ============================================================


def validate_response(
    text,
    triage_decision,
    pulpal_label,
    clinical_structured=None,
    vision_json=None,
):
  violations = []
  vision_json = vision_json or {}
  text_lower = text.lower()

  cold_positive = clinical_structured.get("cold_response") is True
  vitality_terms = [
      "confirms pulp vitality",
      "indicates pulp vitality",
      "demonstrates pulp vitality",
      "proves pulp vitality",
      "confirms a vital pulp",
      "indicates a vital pulp",
  ]
  if cold_positive:
    for term in vitality_terms:
      if term in text_lower:
        violations.append(
            {"type": "COLD_TEST_VITALITY_OVERCLAIM", "text": term}
        )

  pai_error_terms = [
      "probability of inflammation",
      "probability of infection",
      "probability of pulp necrosis",
      "likelihood of inflammation",
      "indicates pulp necrosis",
      "confirms pulp necrosis",
  ]
  for term in pai_error_terms:
    if term in text_lower:
      violations.append({"type": "PAI_CLINICAL_PROBABILITY_MISUSE", "text": term})

  return {
      "flagged": bool(violations),
      "violations": violations,
      "sentences": [v["text"] for v in violations],
  }


# ============================================================
# FINAL DIAGNOSTIC REASONING
# ============================================================


def generate_final_diagnosis(
    model,
    processor,
    vision_json,
    opgagent_json,
    clinical_history,
    clinical_structured,
    evidence,
    diagnostic_consistency,
    triage_decision,
    non_endodontic_assessment,
    escalation,
):
  target_context = opgagent_json.get("target_context", {})
  opgagent_text = json.dumps(
      {
          "target_teeth": target_context.get("target_teeth", []),
          "targeted_findings": target_context.get("targeted_findings", []),
          "frozen_target_findings": target_context.get(
              "frozen_target_findings", []
          ),
      },
      indent=2,
      ensure_ascii=False,
  )

  llm_evidence_text = []
  for item in evidence:
    clean_text = re.sub(
        r"Recommended LLM action:\s*.+?(?=\n|$)",
        "",
        item["text"],
        flags=re.IGNORECASE,
    ).strip()
    llm_evidence_text.append(
        f"[Evidence {item['rank']}] (retrieval score"
        f" {item['score']:.3f})\nSource: {item['source_file']}\nPage:"
        f" {item['page_start']}-{item['page_end']}\nChunk:"
        f" {item['chunk_id']}\n{clean_text}"
    )

  if triage_decision.get("category") == "NON_ENDODONTIC_RED_FLAG":
    llm_context = (
        "[Red-flag evidence summary]\nRetrieved literature supports"
        " consideration of non-endodontic periapical lesion patterns or mimics"
        " when clinical findings do not support a definitive endodontic"
        " diagnosis. Specific rare entities are withheld; retrieval is"
        " supporting evidence only."
    )
  else:
    llm_context = "\n\n".join(llm_evidence_text)

  vision_text = json.dumps(
      {
          "findings": {
              "lesion_count": (
                  vision_json.get("findings", {}).get("lesion_count", 0)
              ),
              "lesions": vision_json.get("findings", {}).get("lesions", []),
          }
      },
      indent=2,
      ensure_ascii=False,
  )

  rule_text = json.dumps(diagnostic_consistency, indent=2, ensure_ascii=False)
  labels = select_diagnostic_labels(
      diagnostic_consistency,
      authoritative_origin_category=triage_decision.get(
          "lesion_origin_classification", {}
      ).get("category"),
  )
  label_text = json.dumps(labels, indent=2, ensure_ascii=False)
  triage_text = json.dumps(triage_decision, indent=2, ensure_ascii=False)
  red_flag_text = json.dumps(
      non_endodontic_assessment, indent=2, ensure_ascii=False
  )
  escalation_text = json.dumps(escalation, indent=2, ensure_ascii=False)
  dynamic_clarifications = build_dynamic_clarifications(clinical_structured)

  target_lock = json.dumps(
      opgagent_json.get("target_context", {}), indent=2, ensure_ascii=False
  )

  # ============================================================
  # STAGE 7.3 — UNIFIED CLINICAL + RADIOGRAPHIC DIAGNOSTIC CONTEXT
  # ============================================================
  # Stage 7.2 remains authoritative and frozen.
  # Stage 7.3 integrates clinician-supplied history/procedural facts
  # with tooth-specific radiographic findings before LLM reasoning.
  # The OPG/vision pipeline is NOT assumed to detect post placement
  # or previous root-canal treatment unless explicitly represented
  # by the clinician in the structured history.

  stage7_3_origin = triage_decision.get(
      "lesion_origin_classification",
      {}
  )

  stage7_3_context = {
      "stage": "7.3",
      "purpose": "Unified clinical-history and tooth-specific radiographic context",
      "selected_tooth_context": opgagent_json.get("target_context", {}),
      "clinical_history_verbatim": clinical_history,
      "clinical_structured": clinical_structured,
      "radiographic_findings_for_selected_tooth": {
          "vision_findings": vision_json.get("findings", {}),
          "opgagent_findings": opgagent_json.get(
              "target_context",
              {}
          ),
      },
      "stage7_2_authoritative_classification": stage7_3_origin,
      "constraints": [
          "Clinician-supplied procedural/history facts are valid clinical facts.",
          "Previous root-canal treatment and post placement may be supplied by the clinician and must not be assumed to be OPG-agent detections.",
          "Do not infer the symptomatic tooth from unrelated OPG findings.",
          "Do not convert a radiographic periapical lesion into a symptomatic diagnosis by itself.",
          "Do not override the Stage 7.2 deterministic origin classification.",
          "Use retrieved literature only as supporting evidence for the established clinical context.",
      ],
  }

  stage7_3_context_text = json.dumps(
      stage7_3_context,
      indent=2,
      ensure_ascii=False,
  )

  print("\n" + "=" * 90)
  print("STAGE 7.3 — UNIFIED CLINICAL + RADIOGRAPHIC DIAGNOSTIC CONTEXT")
  print("=" * 90)
  print(stage7_3_context_text)

  prompt = f"""
You are an Endodontic clinical reasoning assistant.

============================================================
HARD TARGET-TOOTH LOCK & CLINICAL GROUNDING — AUTHORITATIVE
============================================================

TARGET TOOTH:
{target_lock}

STRUCTURED CLINICAL FINDINGS (VERIFIED):
{json.dumps(clinical_structured, indent=2, ensure_ascii=False)}

STAGE 7.3 — UNIFIED CLINICAL + RADIOGRAPHIC CONTEXT:
{stage7_3_context_text}

PROGRAMMATIC DIAGNOSTIC LABELS:
{label_text}

DETERMINISTIC TRIAGE & RED FLAGS:
{triage_text}
{red_flag_text}
{escalation_text}

STAGE 7.2 UNIFIED LESION-ORIGIN CLASSIFICATION:
{json.dumps(triage_decision.get("lesion_origin_classification", {}), indent=2, ensure_ascii=False)}

RADIOGRAPHIC FINDINGS:
{vision_text}
{opgagent_text}

RETRIEVED AUTHORITATIVE EVIDENCE:
{llm_context}

============================================================
CRITICAL REASONING & EVIDENCE RULES (v65)
============================================================
1. DO NOT CONTRADICT RETRIEVED EVIDENCE: When retrieved evidence explicitly identifies pain on biting, mastication pressure, or release tenderness as a feature of symptomatic apical periodontitis or structural tooth cracks, you MUST NOT claim that biting pain is "atypical", "contradictory", or "inconsistent" with an apical diagnosis.
2. PRESERVE STRUCTURED CLINICAL FACTS: Pain on biting ({clinical_structured.get('pain_on_biting')}), pain on release ({clinical_structured.get('pain_on_release')}), and isolated pocketing ({clinical_structured.get('isolated_deep_pocket')}) are ground-truth facts. Do not claim they were unassessed or absent.
3. EXPLANATORY ONLY: You are NOT the primary diagnostic decision maker. Do not alter, override, or upgrade the programmatic diagnostic status.
4. SPECIFIC NON-ENDODONTIC ENTITIES: Do not invent rare non-endodontic neoplasm names for this patient. Refer to them generically as "non-endodontic or fibro-osseous patterns".

CASE CLARIFICATIONS:
{dynamic_clarifications}

============================================================
MANDATORY EVIDENCE AND REASONING RULES
============================================================

The PROGRAMMATIC DIAGNOSTIC LABELS, DETERMINISTIC TRIAGE & RED FLAGS,
and ESCALATION information provided above are authoritative.

STAGE 7.3 DIAGNOSTIC HIERARCHY:
- The Stage 7.2 deterministic classification is authoritative.
- If the deterministic category is VRF_SUSPECTED, VRF/structural
  integrity concern is the PRIMARY TRIAGE CONCERN.
- Do NOT describe symptomatic apical periodontitis or another
  programmatic consistency finding as a "secondary diagnosis"
  unless independently established by the supplied clinical evidence.
- Programmatic diagnostic consistency may be described as a
  compatible/supporting observation, but must not compete with,
  replace, or override the Stage 7.2 classification.
- A periapical lesion alone does not establish symptomatic apical
  periodontitis.
- Do not introduce a diagnosis that is not established by the supplied
  clinical and radiographic evidence.

1. STATUS-AWARE INTERPRETATION
- SUPPORTED findings may be described as supporting evidence.
- CONTRADICTED findings must NEVER be described as supporting,
  consistent, explanatory, or confirmatory evidence.
- When a diagnosis is CONTRADICTED, explicitly state that it is
  contradicted by the relevant clinical/radiographic finding.
- INDETERMINATE findings must remain uncertain and must not be presented
  as established.

2. DO NOT USE CONTRADICTED DIAGNOSES AS EVIDENCE
Do not use any CONTRADICTED diagnosis to justify the final interpretation.
For example, if Localized Asymptomatic Apical Periodontitis is marked
CONTRADICTED because symptoms are present, do NOT state that it supports
the case. State that it is contradicted by the reported symptoms.

3. CLINICAL FINDINGS
Pain on biting, pain on release, an isolated deep periodontal pocket,
and suspected crack/root fracture are clinically relevant findings.
Do not dismiss pain on biting or pain on release merely because the
tooth was previously root canal treated.

4. STRUCTURAL TRIAGE
If the deterministic triage category is
STRUCTURAL_INTEGRITY_COMPROMISE, preserve that category and its basis.
Do not convert it into NON_ENDODONTIC_RED_FLAG unless the deterministic
triage explicitly identifies biological discordance.

5. STAGE 7.3 VRF PRIORITY
If the deterministic triage category is VRF_SUSPECTED:
- Preserve VRF as the primary triage concern.
- State that specialist structural assessment is required.
- Do not call symptomatic apical periodontitis a "secondary diagnosis"
  unless independently established by the supplied evidence.
- Describe compatible apical findings as clinical/radiographic
  consistency observations only.

6. CLINICAL EVIDENCE BOUNDARY
- Do not invent percussion, palpation, tenderness, release pain,
  swelling, sinus tract, or other findings that are not explicitly
  supplied in STRUCTURED CLINICAL FINDINGS.
- Do not introduce patient age, medical history, or other contextual
  factors unless they are explicitly present in the current case
  context.
- Do not use age or unrelated medical history to justify systemic
  evaluation, biopsy, or other escalation.

7. ADVANCED IMAGING
Follow the ESCALATION status exactly.
If advanced imaging or CBCT is INDETERMINATE, do not state that CBCT is
required, crucial, necessary, or automatically indicated. Use conditional
wording such as:
"CBCT may be considered if clinically indicated after clinical
examination and structural assessment."
Only describe CBCT as required when the programmatic escalation explicitly
states that it is required.

6. RADIOGRAPHIC INDEX VS CLINICAL DIAGNOSIS
Do not equate a radiographic periapical index score such as PAI 3 with
"symptomatic apical periodontitis" by itself. A PAI score describes the
radiographic/periapical finding. Symptomatic classification must be
grounded in the verified clinical findings and diagnostic consistency.
Therefore, describe PAI 3 as radiographic evidence of a periapical lesion
or apical disease, and separately explain the clinical evidence supporting
the symptomatic classification.

7. TARGET-TOOTH GROUNDING
All diagnostic reasoning must remain specific to the confirmed TARGET
TOOTH. Do not use unrelated teeth or regional anatomical findings as
evidence for the diagnosis.

8. REASONING
Base the explanation primarily on SUPPORTED findings and verified
clinical findings. Important CONTRADICTED findings should be explicitly
acknowledged as contradicted and must not be used as positive evidence.

9. STRUCTURAL TRIAGE PRIORITY
When the deterministic triage category is STRUCTURAL_INTEGRITY_COMPROMISE,
the reasoning must prioritize the structural findings (pain on release,
isolated deep periodontal pocket, and crack/root-fracture suspicion) as
the primary triage concern. A supported diagnosis of symptomatic apical
periodontitis may be reported as a concurrent supported apical diagnosis,
but it must not be presented as the primary explanation that supersedes
or resolves the structural-integrity concern. Do not state that
symptomatic apical periodontitis explains the case without explicitly
preserving the structural differential.

7.2. DETERMINISTIC CLINICAL ORIGIN GATE
The lesion-origin classification supplied by the programmatic Stage 7.2
classifier is authoritative and priority ordered:

GATE 1 — VRF:
Previously treated tooth AND isolated narrow deep pocket OR
coronal/lateral sinus tract OR J-shaped/halo lesion morphology
=> VRF_SUSPECTED.

GATE 2 — CRACKED TOOTH:
Only when Gate 1 is negative:
(pain on release AND difficulty localizing pain) OR visible crack line
=> CRACKED_TOOTH_SUSPECTED.

GATE 3 — STANDARD ENDODONTIC ORIGIN:
Only when Gates 1 and 2 are negative:
cold response == False AND EPT response == False
=> ENDODONTIC_ORIGIN_SUPPORTED.

GATE 4 — NON-ENDODONTIC RED FLAG:
Only when Gates 1–3 are negative:
cold response == True OR EPT response == True,
AND a target-associated periapical lesion is present
=> NON_ENDODONTIC_RED_FLAG.

Otherwise => INDETERMINATE.

Do not independently reorder, override, or replace this classification.
Unknown/None clinical findings do not count as negative findings.

10. PAI / RADIOGRAPHIC INDEX TERMINOLOGY
Never state that PAI 3, by itself, indicates or establishes
symptomatic apical periodontitis. Describe PAI 3 strictly as a
radiographic/periapical index finding or periapical lesion finding.

When retrieved evidence uses terminology that equates a PAI category
with a clinical diagnostic label, do not reproduce that equivalence as
an independent diagnostic conclusion. The clinical diagnosis must remain
grounded in the verified clinical findings and diagnostic-consistency
module.

When deterministic triage identifies STRUCTURAL_INTEGRITY_COMPROMISE,
the presence of a PAI 3 lesion may support a concurrent apical diagnosis
when the clinical findings support it, but it must not be presented as
the primary explanation for the patient's complaint or as superseding
the structural-integrity concern.

============================================================

============================================================
YOUR TASK
============================================================
Provide a concise, clinically grounded explanation using EXACTLY these headings in this order:

1. Triage conclusion
2. Radiographic findings
3. Diagnostic reasoning
4. Evidence supporting the interpretation
5. Missing or uncertain information
"""

  raw_answer = qwen_text(model, processor, prompt, max_new_tokens=900)
  # ========================================================
  # STAGE 7.3 — FINAL EVIDENCE-BOUNDARY SANITIZATION
  # ========================================================

  # 1. Generic Typo and Grammar Cleanup
  raw_answer = re.sub(r"\bCrack/Rot\b", "Crack/Root", raw_answer, flags=re.IGNORECASE)
  raw_answer = re.sub(r"\ba\s+the\s+Stage\s+7\.2\b", "the Stage 7.2", raw_answer, flags=re.IGNORECASE)
  raw_answer = re.sub(r"\b(suspected\s+)?crack/root\s+fracture\b", "the Stage 7.2 VRF_SUSPECTED classification", raw_answer, flags=re.IGNORECASE)
  raw_answer = re.sub(r"\bpossible\s+crack\s+or\s+root\s+fracture\b", "the Stage 7.2 VRF_SUSPECTED classification", raw_answer, flags=re.IGNORECASE)

  # 2. VRF_SUSPECTED Triage Priority Enforcement
  if triage_decision.get("category") == "VRF_SUSPECTED":
      raw_answer = re.sub(
          r"Localized\s+symptomatic\s+apical\s+periodontitis\s+is\s+defined[^\.\n]*[\.\n]",
          "Stage 7.2 classified the case as VRF_SUSPECTED based on prior endodontic treatment and structural findings. ",
          raw_answer,
          flags=re.IGNORECASE
      )

      raw_answer = re.sub(
          r"PAI\s+Class\s+3\s+is\s+indicative\s+of\s+a\s+periapical\s+lesion,\s+supporting\s+the\s+radiographic\s+findings\s+of\s+symptomatic\s+apical\s+periodontitis[\.]?",
          "PAI Class 3 indicates a radiographic periapical lesion, which is an observational finding that does not override the primary Stage 7.2 VRF_SUSPECTED classification.",
          raw_answer,
          flags=re.IGNORECASE
      )

      raw_answer = re.sub(
          r"The\s+presence\s+of\s+a\s+periapical\s+radiolucency\s+\(PAI\s+Class\s+3\)\s+supports\s+the\s+diagnosis\s+of\s+symptomatic\s+apical\s+periodontitis[\.]?",
          "The periapical lesion (PAI Class 3) is a radiographic finding and does not override the Stage 7.2 VRF_SUSPECTED classification.",
          raw_answer,
          flags=re.IGNORECASE
      )

      raw_answer = re.sub(
          r"supports\s+the\s+diagnosis\s+of\s+symptomatic\s+apical\s+periodontitis[\.]?",
          "is compatible with apical inflammatory findings but does not override the Stage 7.2 VRF_SUSPECTED classification.",
          raw_answer,
          flags=re.IGNORECASE
      )

      raw_answer = re.sub(
          r"The\s+patient\s+reports\s+pain\s+on\s+biting,\s+which\s+is\s+a\s+(?:classic\s+symptom|hallmark)\s+of\s+symptomatic\s+apical\s+periodontitis[\.]?",
          "Pain on biting is present and is compatible with periapical inflammation; however, Stage 7.2 VRF_SUSPECTED remains the primary triage classification.",
          raw_answer,
          flags=re.IGNORECASE
      )

  # 3. Clinical Testing Boundaries (Percussion & Palpation unsupplied)
  if (
      clinical_structured.get("percussion") is None
      and clinical_structured.get("palpation") is None
      and clinical_structured.get("pain_on_biting") is True
  ):
      raw_answer = re.sub(
          r"Pain\s+or\s+tenderness\s+was\s+reported\s+with\s+biting,\s+percussion,\s+and/or\s+palpation[\.]?",
          "Pain on biting was reported. Percussion and palpation findings were not supplied.",
          raw_answer,
          flags=re.IGNORECASE
      )

  # 4. Systemic Status Boundary
  if clinical_structured.get("systemic_infection_signs") is None:
      raw_answer = re.sub(
          r"Although\s+not\s+explicitly\s+stated,\s+the\s+presence\s+of\s+a\s+periapical\s+lesion\s+raises\s+the\s+possibility\s+of\s+systemic\s+involvement[^\.\n]*[\.\n]",
          "Systemic infection status was not assessed/provided. ",
          raw_answer,
          flags=re.IGNORECASE
      )
      raw_answer = re.sub(
          r"There\s+is\s+no\s+indication\s+of\s+systemic\s+infection\s+or\s+involvement[\.]?",
          "Systemic infection status was not assessed/provided.",
          raw_answer,
          flags=re.IGNORECASE
      )
      raw_answer = re.sub(
          r"a\s+thorough\s+systemic\s+evaluation\s+would\s+be\s+prudent",
          "systemic infection status was not assessed/provided",
          raw_answer,
          flags=re.IGNORECASE
      )

  labels_for_validation = select_diagnostic_labels(diagnostic_consistency)

  validation = validate_response(
      raw_answer,
      triage_decision,
      labels_for_validation["probable_pulpal_diagnosis"],
      clinical_structured=clinical_structured,
      vision_json=vision_json,
  )

  answer = raw_answer
  if validation["flagged"]:
    caveat = (
        "\n\nAUTOMATED SAFETY NOTE: The deterministic triage layer has NOT"
        " established complete pulpal necrosis. Statements should be evaluated"
        " cautiously."
    )
    answer = raw_answer.rstrip() + caveat

  evidence_lines = ["", "Evidence sources:"]
  for item in evidence:
    evidence_lines.append(
        f"[Evidence {item['rank']}] {item['source_file']}, p."
        f" {item['page_start']}-{item['page_end']} (retrieval score"
        f" {item['score']:.4f})"
    )

  evidence_lines.extend([
      "",
      "Specialist verification disclaimer:",
      (
          "This is an AI-assisted probable diagnostic interpretation and must"
          " be clinically verified by a qualified dental"
          " specialist/endodontist."
      ),
  ])

  final_text = answer.rstrip() + "\n" + "\n".join(evidence_lines)
  return {
      "final_response": final_text,
      "raw_model_response": raw_answer,
      "safety_validation": validation,
  }


# ============================================================
# PROGRAMMATIC DIAGNOSTIC LABELS
# ============================================================


def select_diagnostic_labels(
    diagnostic_consistency,
    authoritative_origin_category=None,
):
  apical = diagnostic_consistency.get("apical_consistency", {})
  pulpal = diagnostic_consistency.get("pulpal_consistency", {})

  supported_apical = [
      k for k, v in apical.items() if v.get("status") in {"supported", "compatible"}
  ]
  contradicted_apical = [
      k for k, v in apical.items() if v.get("status") == "contradicted"
  ]

  systemic_category = "apical_periodontitis_with_systemic_involvement"
  sinus_category = "localized_apical_periodontitis_with_sinus_tract"

  # Stage 7.2 hierarchy:
  # A deterministic VRF_SUSPECTED classification is authoritative and must
  # not be overridden by a generic apical-periodontitis consistency label.
  if authoritative_origin_category == "VRF_SUSPECTED":
    apical_value = None
    apical_status = "indeterminate"
  elif systemic_category in supported_apical:
    apical_value = systemic_category
    apical_status = "supported"
  elif sinus_category in supported_apical:
    apical_value = sinus_category
    apical_status = "supported"
  elif len(supported_apical) == 1:
    apical_value = supported_apical[0]
    apical_status = "supported"
  else:
    apical_value = None
    apical_status = "indeterminate"

  pulpal_candidates = [
      (name, value.get("status"))
      for name, value in pulpal.items()
      if value.get("status") in {"supported", "suggestive_but_incomplete"}
  ]

  if len(pulpal_candidates) == 1:
    pulpal_value, pulpal_status = pulpal_candidates[0]
  else:
    pulpal_value = None
    pulpal_status = "indeterminate"

  return {
      "probable_pulpal_diagnosis": {
          "value": pulpal_value,
          "status": pulpal_status,
      },
      "probable_apical_diagnosis": {
          "value": apical_value,
          "status": apical_status,
      },
      "contradicted_apical_categories": contradicted_apical,
      "supported_apical_categories": supported_apical,
  }


def build_structured_output(
    vision_json,
    opgagent_json,
    clinical_history,
    clinical_structured,
    diagnostic_consistency,
    diagnostic_labels,
    non_endodontic_assessment,
    triage_decision,
    escalation,
    origin_classification,
    final_reasoning,
    evidence,
    raw_model_response=None,
    safety_validation=None,
):
  evidence_refs = [{
      "id": f"Evidence {item['rank']}",
      "source": item["source_file"],
      "pages": f"{item['page_start']}-{item['page_end']}",
      "chunk_id": item["chunk_id"],
      "retrieval_score": round(item["score"], 6),
  } for item in evidence]

  pulpal = diagnostic_labels["probable_pulpal_diagnosis"]
  apical = diagnostic_labels["probable_apical_diagnosis"]

  uncertainties = []
  if pulpal["status"] in {"suggestive_but_incomplete", "indeterminate"}:
    uncertainties.append(
        "Pulpal diagnosis is not fully established from clinical findings."
    )

  for name, result in (
      diagnostic_consistency.get("apical_consistency", {}).items()
  ):
    if result.get("status") == "contradicted":
      uncertainties.append(
          f"{name} was explicitly contradicted by clinical findings."
      )

  return {
      "system": {
          "pipeline": "Phase 4.5 Interactive Endodontic MLLM (v65) — Stage 7.2",
          "vision_detector": "FasterRCNN_epoch14",
          "vision_classifier": "ResNet50_epoch12",
          "terminology_framework": "2025 AAE/ESE periapical terminology",
      },
      "radiographic_findings": vision_json,
      "opgagent_findings": opgagent_json,
      "clinical_history": clinical_history,
      "clinical_structured": clinical_structured,
      "diagnostic_consistency": diagnostic_consistency,
      "probable_diagnosis": {
          "pulpal": {"value": pulpal["value"], "status": pulpal["status"]},
          "apical": {"value": apical["value"], "status": apical["status"]},
      },
      "non_endodontic_assessment": non_endodontic_assessment,
      "triage_decision": triage_decision,
      "lesion_origin_classification": (
          origin_classification
          if origin_classification is not None
          else {}
      ),
      "escalation": escalation,
      "reasoning": final_reasoning,
      "raw_model_response": raw_model_response,
      "safety_validation": safety_validation
      or {"flagged": False, "sentences": []},
      "uncertainties": uncertainties,
      "evidence": evidence_refs,
      "specialist_verification_required": True,
  }


# ============================================================
# MAIN
# ============================================================


def main():
  parser = argparse.ArgumentParser(
      description="Phase 4 Interactive Endodontic MLLM (v65)"
  )
  parser.add_argument("image", help="Path to a dental radiograph")
  parser.add_argument(
      "--top-k",
      type=int,
      default=5,
      help="Number of retrieved evidence chunks",
  )
  args = parser.parse_args()

  image_path = Path(args.image).expanduser().resolve()
  if not image_path.exists():
    print(f"ERROR: Image not found: {image_path}")
    sys.exit(1)

  start_time = time.time()
  print("\nINPUT IMAGE:", image_path)

  # STEP 1 — Frozen vision
  print("\n" + "=" * 90)
  print("STEP 1 — FROZEN RADIOGRAPHIC VISION")
  print("=" * 90)
  vision_json = analyze_radiograph(str(image_path))
  print(json.dumps(vision_json, indent=2))

  # STEP 1B — OPGAgent
  print("\n" + "=" * 90)
  print("STEP 1B — OPGAGENT PERIAPICAL VISUAL EVIDENCE")
  print("=" * 90)
  opgagent_json = run_opgagent(image_path)

  # INTERACTIVE GATE
  target_context = interactive_target_tooth_gate(opgagent_json, vision_json)
  if not target_context.get("confirmed", False):
    print("\nTarget tooth not confirmed. Aborting pipeline.")
    return

  full_vision_json = vision_json
  full_opgagent_json = opgagent_json
  opgagent_json["target_context"] = target_context

  target_evidence = build_target_specific_evidence(
      full_vision_json, full_opgagent_json, target_context
  )
  targeted_vision_json = target_evidence["vision"]
  targeted_opgagent_json = target_evidence["opgagent"]

  combined_visual_evidence = build_combined_visual_evidence(
      targeted_vision_json, targeted_opgagent_json
  )

  # STEP 2 — Clinical History Elicitation
  model, processor = load_qwen()
  question_text = generate_history_questions(
      model, processor, combined_visual_evidence
  )
  clinical_history = collect_history(question_text)

  # STEP 2B — Parsing & Structured Consistency Check (v65 Engine)
  print("\n" + "=" * 90)
  print("STEP 2B — CLINICAL LANGUAGE NORMALIZATION & STRUCTURED EXTRACTION")
  print("=" * 90)
  clinical_structured = extract_clinical_structured(clinical_history)
  print(json.dumps(clinical_structured, indent=2))

  diagnostic_consistency = evaluate_diagnostic_consistency(
      clinical_structured,
      {
          "lesion_count": targeted_vision_json["findings"]["lesion_count"],
          "lesions": targeted_vision_json["findings"]["lesions"],
      },
  )
  print("\n" + "-" * 90)
  print("DIAGNOSTIC CONSISTENCY CHECK")
  print("-" * 90)
  print(json.dumps(diagnostic_consistency, indent=2))

  # STEP 2C — Red-Flag & Precedence Assessment
  # Pre-triage labels are used only by the legacy red-flag assessment.
  # Stage 7.2 authoritative VRF classification is applied afterward.
  pretriage_diagnostic_labels = select_diagnostic_labels(
      diagnostic_consistency
  )
  pretriage_endodontic_consistency = {
      "pulpal": pretriage_diagnostic_labels["probable_pulpal_diagnosis"],
      "apical": pretriage_diagnostic_labels["probable_apical_diagnosis"],
  }

  red_flag_result = assess_non_endodontic_red_flags(
      radiographic_findings=targeted_vision_json.get("findings", {}),
      clinical_history=clinical_structured,
      endodontic_consistency=pretriage_endodontic_consistency,
  )

  discordance = assess_periapical_discordance(
      vision_json=targeted_vision_json,
      opgagent_json=targeted_opgagent_json,
      clinical_structured=clinical_structured,
  )

  # ==========================================================
  # STAGE 7.2 — UNIFIED PRIORITY-ORDERED CLINICAL ORIGIN GATE
  # ==========================================================
  origin_classification = classify_lesion_origin(
      clinical_structured=clinical_structured,
      vision_json=targeted_vision_json,
  )

  print("\n" + "-" * 90)
  print("STAGE 7.2 — UNIFIED CLINICAL ORIGIN CLASSIFICATION")
  print("-" * 90)
  print(json.dumps(origin_classification, indent=2))

  red_flag_result = merge_red_flag_assessment(
      red_flag_result,
      discordance,
      clinical_structured,
      origin_classification=origin_classification,
  )

  non_endodontic_assessment = red_flag_result["non_endodontic_assessment"]
  triage_decision = red_flag_result["triage_decision"]
  escalation = red_flag_result["escalation"]

  diagnostic_labels = select_diagnostic_labels(
      diagnostic_consistency,
      authoritative_origin_category=triage_decision.get(
          "lesion_origin_classification", {}
      ).get("category"),
  )
  endodontic_consistency_for_triage = {
      "pulpal": diagnostic_labels["probable_pulpal_diagnosis"],
      "apical": diagnostic_labels["probable_apical_diagnosis"],
  }

  # ==========================================================
  # STAGE 7.3 — AUTHORITATIVE ESCALATION ALIGNMENT
  # Stage 7.2 classification is authoritative. A suspected VRF
  # requires specialist structural assessment regardless of the
  # legacy red-flag escalation state.
  # ==========================================================
  if triage_decision.get("category") == "VRF_SUSPECTED":
    escalation["specialist_review"] = True
    escalation["advanced_imaging"] = (
        "consider_cbct_when_clinically_indicated"
    )
    escalation["definitive_pathologic_diagnosis_required"] = False

  non_endodontic_case = triage_decision["category"] == "NON_ENDODONTIC_RED_FLAG"

  # STEP 3 — Dynamic RAG
  print("\n" + "=" * 90)
  print(
      "STEP 3 — "
      + (
          "NON-ENDODONTIC RED-FLAG RETRIEVAL"
          if non_endodontic_case
          else "ENDODONTIC KNOWLEDGE RETRIEVAL"
      )
  )
  print("=" * 90)

  retrieval_query = build_retrieval_query(
      targeted_vision_json,
      clinical_history,
      clinical_structured=clinical_structured,
      non_endodontic=non_endodontic_case,
      opgagent_json=targeted_opgagent_json,
      target_context=target_context,
  )
  retriever, embeddings, metadata, chunks = load_rag_index(
      merged=non_endodontic_case
  )
  evidence = retrieve_evidence(
      retriever,
      embeddings,
      metadata,
      chunks,
      retrieval_query,
      top_k=args.top_k,
  )

  # STEP 4 — Reasoning & Explanation Generation
  print("\n" + "=" * 90)
  print("STEP 4 — EVIDENCE-GROUNDED REASONING & GENERATION")
  print("=" * 90)

  diagnosis_result = generate_final_diagnosis(
      model,
      processor,
      targeted_vision_json,
      targeted_opgagent_json,
      clinical_history,
      clinical_structured,
      evidence,
      diagnostic_consistency,
      triage_decision,
      non_endodontic_assessment,
      escalation,
  )

  final_response = diagnosis_result["final_response"]
  raw_model_response = diagnosis_result["raw_model_response"]
  safety_validation = diagnosis_result["safety_validation"]

  safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
  structured_output = build_structured_output(
      vision_json=vision_json,
      opgagent_json=opgagent_json,
      clinical_history=clinical_history,
      clinical_structured=clinical_structured,
      diagnostic_consistency=diagnostic_consistency,
      diagnostic_labels=diagnostic_labels,
      non_endodontic_assessment=non_endodontic_assessment,
      triage_decision=triage_decision,
      escalation=escalation,
      origin_classification=origin_classification,
      final_reasoning=final_response,
      evidence=evidence,
      raw_model_response=raw_model_response,
      safety_validation=safety_validation,
  )

  structured_output_file = (
      OUTPUT_DIR / f"{safe_stem}_phase4_structured_diagnostic.json"
  )
  with structured_output_file.open("w", encoding="utf-8") as f:
    json.dump(structured_output, f, indent=2, ensure_ascii=False)
  print(f"\nAUDIT OUTPUT SAVED TO: {structured_output_file}")

  print("\n" + final_response)
  print("\n" + "=" * 90)
  print(f"PHASE 4 COMPLETE (Runtime: {time.time() - start_time:.1f}s)")
  print("=" * 90)


if __name__ == "__main__":
  main()