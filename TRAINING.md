# ENDOLLM Training and Model Development Documentation

## Overview

This document describes the model-development and training status of the **ENDOLLM** system associated with the current manuscript/repository implementation.

The current public implementation is:

```text
ENDOLLM
Phase 4.5 Interactive Endodontic MLLM
Version: v65
Project stage: 7/9
Diagnostic-rule stage: 7.2
```

A critical distinction should be made between **model training** and the **v65 multimodal inference/integration pipeline**.

The current v65 script does not itself train a neural network.

Instead, v65 integrates previously developed/frozen visual models, external visual-analysis components, a pretrained multimodal language model, and a pretrained retrieval encoder into an interactive endodontic diagnostic workflow.

---

# 1. Training vs Inference

The v65 implementation is primarily an **inference and orchestration layer**.

Its principal functions are:

1. loading pretrained/frozen models;
2. processing a dental radiograph;
3. obtaining radiographic observations;
4. obtaining tooth-level visual findings;
5. obtaining clinician-confirmed target-tooth information;
6. collecting structured clinical history;
7. applying deterministic diagnostic rules;
8. retrieving supporting evidence;
9. generating evidence-grounded reasoning;
10. validating the generated response;
11. producing an auditable structured JSON output.

The v65 script does **not** perform model parameter optimization.

There is no training loop containing operations such as:

```text
forward pass
loss calculation
backpropagation
optimizer step
gradient update
epoch iteration
checkpoint saving
```

Accordingly, v65 should be described as the **deployment/inference integration implementation of the ENDOLLM architecture**, rather than as the script used to train every underlying model.

---

# 2. Model Components

The current v65 system contains several distinct model/component classes.

| Component                  | Model / implementation      | Role in v65                           | Status in v65                                |
| -------------------------- | --------------------------- | ------------------------------------- | -------------------------------------------- |
| Periapical lesion detector | FasterRCNN_epoch14          | Radiographic lesion detection         | Frozen                                       |
| Radiographic classifier    | ResNet50_epoch12            | Radiographic classification           | Frozen                                       |
| Tooth enumeration          | OPGAgent YOLO model         | Tooth identification                  | External pretrained checkpoint               |
| OPGAgent visual experts    | TVEM / MaskDINO checkpoints | Additional visual findings            | External pretrained checkpoints              |
| Multimodal LLM             | Qwen/Qwen2.5-VL-7B-Instruct | History elicitation and reasoning     | Pretrained base model; inference only in v65 |
| Retrieval encoder          | BAAI/bge-small-en-v1.5      | Knowledge retrieval                   | Pretrained encoder; inference only           |
| Diagnostic rules           | ENDOLLM programmatic rules  | Deterministic diagnostic/triage logic | Rule-based, not learned                      |

The v65 code explicitly identifies the Faster R-CNN and ResNet-50 pipeline as the original frozen pipeline and states that the OPGAgent layer is additive.

---

# 3. Frozen Radiographic Vision Models

The core ENDOLLM radiographic pipeline used by v65 contains:

```text
FasterRCNN_epoch14
ResNet50_epoch12
```

The v65 structured output records these models explicitly as:

```text
vision_detector:
    FasterRCNN_epoch14

vision_classifier:
    ResNet50_epoch12
```

The associated terminology framework recorded by the system is:

```text
2025 AAE/ESE periapical terminology
```

These models are treated as **frozen components** during the v65 workflow.

The v65 implementation does not modify their parameters.

The code imports the existing vision pipeline through:

```python
from vision_pipeline import analyze_radiograph
```

and calls:

```python
vision_json = analyze_radiograph(str(image_path))
```

during inference.

---

# 4. Original Vision Pipeline and v65

The v65 implementation describes the OPGAgent component as an additive visual-evidence layer.

The code explicitly states:

```text
OPGAgent is an additive visual-evidence layer.
The original v62 FasterRCNN/ResNet50 pipeline remains untouched.
```

Therefore, the architecture should be understood as:

```text
Original frozen radiographic pipeline
                +
        OPGAgent visual layer
                +
      clinical interaction
                +
    deterministic diagnostic rules
                +
             RAG
                +
           Qwen2.5-VL
                =
             ENDOLLM v65
```

This is an important distinction for reproducibility because v65 does not represent a newly retrained Faster R-CNN or ResNet-50 model.

---

# 5. OPGAgent Models

The v65 implementation incorporates OPGAgent as an additional visual-analysis layer.

The code references a separate OPGAgent environment and model directory:

```text
OPGAgent/
opgagent_env/
```

The tooth-enumeration model is:

```text
api_service/yolo_enumeration/model/best.pt
```

The configured OPGAgent components include models for:

```text
11diseases
4quadrants
mandibular_maxillary
bone_loss
```

The corresponding TVEM/MaskDINO checkpoints are referenced by the v65 implementation.

These models are loaded as external pretrained checkpoints.

They are not trained by the v65 script.

---

# 6. OPGAgent Tooth Enumeration

The OPGAgent tooth-enumeration model provides tooth-level localization and identification.

The current v65 implementation uses a confidence threshold of:

```text
0.25
```

The resulting detections are converted into human-readable tooth names.

The system then uses these tooth-level observations to support the interactive target-tooth selection process.

The target tooth is ultimately identified and confirmed by the clinician rather than inferred solely from the model output.

The code explicitly presents OPGAgent findings as observations and warns that the presence of a radiographic finding does not itself establish that it is the cause of the patient's symptoms.

---

# 7. OPGAgent Visual-Expert Models

The current v65 implementation references four OPGAgent TVEM/MaskDINO model configurations:

```text
11diseases
4quadrants
mandibular_maxillary
bone_loss
```

The corresponding checkpoint files referenced in the code include:

```text
Teeth_Visual_Experts_Maskdino_Swinl_x-ray_11diseases.pth

Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_4quadrants.pth

Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.pth

Teeth_Visual_Experts_Maskdino_Swinl_x-ray_bone_loss_1disease.pth
```

These checkpoints are referenced by path and loaded for inference.

They are not retrained inside v65.

---

# 8. Qwen2.5-VL-7B-Instruct

The language-model component used by v65 is:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

The v65 code loads this model using the Hugging Face Transformers interface.

The model is loaded using:

```text
4-bit quantization
NF4 quantization
double quantization
float16 computation
automatic device mapping
```

The model is placed in evaluation mode:

```python
model.eval()
```

The current v65 script does not fine-tune Qwen.

There is no training dataset loader, optimizer, loss function, gradient update, or parameter-update operation for Qwen.
