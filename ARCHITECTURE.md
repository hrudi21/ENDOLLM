# ENDOLLM Architecture

## Phase 4.5 Interactive Endodontic MLLM — v65

**Project:** ENDOLLM
**Overall project stage:** 7/9
**Current diagnostic-rule stage:** 7.2
**Implementation:** Phase 4.5 Interactive Endodontic MLLM (v65)

---

# 1. Architecture Overview

ENDOLLM is a hybrid multimodal system designed for **interactive endodontic diagnostic decision support**.

The current v65 implementation does not depend on a single model to make an autonomous diagnosis.

Instead, it combines:

```text
Radiographic vision
        +
Tooth-level visual analysis
        +
Clinician target-tooth selection
        +
Structured clinical information
        +
Deterministic diagnostic rules
        +
Lesion-origin classification
        +
Retrieval-augmented generation
        +
Multimodal language-model reasoning
        +
Safety validation
        +
Structured audit output
```

The architecture is intentionally modular so that visual perception, clinical decision rules, evidence retrieval, language-model reasoning, and safety validation can be examined separately.

---

# 2. High-Level Architecture

The current v65 pipeline can be represented as:

```text
                         RADIOGRAPH
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Frozen Vision Pipeline      │
              │                             │
              │ FasterRCNN_epoch14         │
              │ ResNet50_epoch12            │
              └──────────────┬──────────────┘
                             │
                             ▼
                  Radiographic observations
                             │
                             │
                  ┌──────────▼──────────┐
                  │       OPGAgent      │
                  │                     │
                  │ Tooth enumeration   │
                  │ Quadrants           │
                  │ Visual findings     │
                  │ Bone loss           │
                  └──────────┬──────────┘
                             │
                             ▼
                  Tooth-level observations
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Interactive Target-Tooth     │
              │ Gate                         │
              │                              │
              │ Clinician selects and       │
              │ confirms symptomatic tooth   │
              └──────────────┬───────────────┘
                             │
                             ▼
                   Target-specific evidence
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Clinical History Elicitation │
              │                              │
              │ Qwen2.5-VL                  │
              │ Maximum 5 focused questions │
              └──────────────┬───────────────┘
                             │
                             ▼
                 Structured clinical profile
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    Diagnostic consistency          Stage 7.2 classifier
              │                             │
              └──────────────┬──────────────┘
                             ▼
                   Authoritative triage
                             │
                             ▼
                 Dynamic retrieval query
                             │
                             ▼
                    BGE retrieval
                             │
                             ▼
                    Evidence chunks
                             │
                             ▼
                 Qwen2.5-VL reasoning
                             │
                             ▼
                    Safety validation
                             │
                             ▼
                 Structured JSON output
```

---

# 3. Architectural Principle: Separation of Responsibilities

The v65 architecture separates system responsibilities.

| Component              | Primary responsibility                               |
| ---------------------- | ---------------------------------------------------- |
| Frozen vision pipeline | Radiographic perception                              |
| OPGAgent               | Additional tooth-level visual evidence               |
| Clinician              | Target-tooth identification and clinical information |
| Structured extraction  | Converts clinical narrative into explicit fields     |
| Diagnostic rules       | Clinical consistency assessment                      |
| Stage 7.2 classifier   | Lesion-origin prioritization                         |
| RAG                    | Evidence retrieval                                   |
| Qwen2.5-VL             | Question generation and evidence-grounded reasoning  |
| Safety validator       | Detects prohibited/inappropriate claims              |
| Structured output      | Auditability and reproducibility                     |

This separation is a core design feature of ENDOLLM.

---

# 4. Module 1 — Frozen Radiographic Vision

The first visual component is the established/frozen radiographic pipeline.

The v65 system identifies:

```text
FasterRCNN_epoch14
ResNet50_epoch12
```

and records:

```text
2025 AAE/ESE periapical terminology
```

as the terminology framework.

The pipeline is imported into v65 through:

```python
from vision_pipeline import analyze_radiograph
```

The radiograph is passed to the frozen pipeline, which returns structured radiographic observations.

The v65 implementation does not retrain these models.

---

# 5. Module 2 — OPGAgent Visual Evidence

OPGAgent is incorporated as an **additional visual-evidence layer**.

Its role includes:

* tooth enumeration;
* quadrant identification;
* additional visual finding detection;
* mandibular/maxillary-related findings;
* bone-loss assessment.

The v65 implementation explicitly preserves the original frozen radiographic pipeline while adding OPGAgent findings.

The architecture can therefore be represented as:

```text
Original frozen vision
        +
OPGAgent
        ↓
Combined visual evidence
```

OPGAgent is not treated as a replacement for the original radiographic pipeline.

---

# 6. Module 3 — Tooth Enumeration

The OPGAgent tooth-enumeration component uses:

```text
api_service/yolo_enumeration/model/best.pt
```

The configured confidence threshold is:

```text
0.25
```

The resulting detections are converted into human-readable tooth identifiers.

This allows the subsequent clinician interaction to operate using recognizable tooth names rather than relying only on bounding-box coordinates.

---

# 7. Module 4 — Visual Expert Models

The v65 implementation references additional OPGAgent visual experts for:

```text
11 diseases
4 quadrants
mandibular/maxillary structures
bone loss
```

The configured TVEM confidence threshold is:

```text
0.30
```

These models generate additional observations that are passed into the target-tooth workflow.

They remain visual observations and do not independently establish the final clinical diagnosis.

---

# 8. Module 5 — Target-Tooth Gate

The **target-tooth gate** is a central architectural component.

A radiograph can contain multiple abnormalities.

Therefore, ENDOLLM does not assume:

```text
radiographic abnormality = symptomatic tooth
```

Instead, the system asks the clinician to identify the symptomatic tooth or teeth.

The workflow is:

```text
Visual findings
       ↓
Human-readable tooth findings
       ↓
Clinician selects symptomatic tooth
       ↓
System validates selection
       ↓
Clinician confirms
       ↓
Target tooth becomes locked
```

Only after this step does the system proceed with target-specific diagnostic reasoning.

---

# 9. Target-Tooth Validation

The system validates the clinician's target-tooth selection against the teeth detected by the visual pipeline.

If the selected tooth cannot be matched to a detected human-readable tooth, the system does not silently substitute another tooth.

The clinician is required to provide a valid target selection.

This prevents downstream reasoning from being anchored to an unverified or incorrectly interpreted tooth identifier.

---

# 10. Module 6 — Target-Specific Evidence

Once the target tooth is confirmed, v65 restricts downstream evidence to findings associated with the selected tooth.

The architecture distinguishes:

```text
Global visual findings
        vs.
Target-specific findings
```

Only target-associated findings are allowed to function as direct diagnostic evidence for the selected tooth.

Other findings can remain contextual but are not automatically attributed to the target tooth.

---

# 11. Spatial Association

The v65 implementation associates findings with teeth using spatial relationships.

The configured parameters include:

```text
Minimum IoU:                0.05
Minimum lesion containment: 0.50
```

The conceptual process is:

```text
Lesion bounding box
        +
Tooth bounding box
        ↓
Spatial association
        ↓
Target-associated finding
```

This provides a programmatic mechanism for linking radiographic findings to individual teeth.

---

# 12. Module 7 — Clinical History Elicitation

After target-tooth selection, Qwen2.5-VL is used to generate focused clinical questions.

The system limits the interaction to a maximum of:

```text
5 questions
```

The questions can address:

* symptoms;
* pain on biting;
* pain on release;
* pain localization;
* cold response;
* cold lingering;
* EPT response;
* percussion;
* palpation;
* swelling;
* sinus tract;
* periodontal probing;
* cracks;
* previous root canal treatment;
* trauma;
* other clinically relevant features.

The LLM is instructed to ask questions rather than directly provide a diagnosis during this phase.

---

# 13. Module 8 — Clinical Information Normalization

The clinician provides the clinical history as narrative text.

The system then normalizes clinical terminology before extracting structured information.

The normalization layer handles common terminology variations and typographical differences.

This allows subsequent rules to operate on standardized clinical concepts.

---

# 14. Module 9 — Structured Clinical Representation

The normalized clinical history is converted into a structured dictionary.

The system uses three-state logic:

```text
True
False
None
```

The third state is important.

It represents:

```text
Unknown / not established
```

rather than:

```text
False
```

This prevents missing clinical information from being interpreted as a negative clinical finding.

---

# 15. Clinical Variables

The structured clinical representation includes fields such as:

```text
tooth_identified
chief_complaint
symptom_onset_duration
pain_on_biting
pain_on_release
difficulty_localizing_pain
cold_response
cold_lingering
ept_response
percussion
palpation
spontaneous_pain
swelling
sinus_tract
sinus_tract_location
isolated_deep_pocket
isolated_narrow_deep_pocket
bleeding_on_probing
mobility
furcation_involvement
crack_suspected
visible_crack_line
vertical_root_fracture_suspected
previously_treated
lesion_morphology
trauma_history
systemic_infection_signs
```

These structured findings become inputs to the deterministic diagnostic layer.

---

# 16. Module 10 — Diagnostic Consistency

ENDOLLM uses programmatic diagnostic-consistency assessment.

The v65 implementation imports:

```python
evaluate_diagnostic_consistency
```

The purpose is to compare structured clinical information with the available diagnostic framework.

This layer is deterministic and is separate from the LLM narrative generation.

---

# 17. Module 11 — Non-Endodontic Red-Flag Assessment

The system separately assesses possible non-endodontic red flags.

The v65 implementation imports:

```python
assess_non_endodontic_red_flags
```

This assessment is preserved as a distinct component rather than allowing the LLM to independently generate rare alternative diagnoses.

The architecture therefore distinguishes:

```text
Clinical/radiographic discordance
        ↓
Programmatic red-flag assessment
```

from:

```text
LLM speculation
```

---

# 18. Module 12 — Stage 7.2 Lesion-Origin Classifier

The current deterministic lesion-origin classifier uses a priority hierarchy.

```text
Priority 1
VRF_SUSPECTED

Priority 2
CRACKED_TOOTH_SUSPECTED

Priority 3
ENDODONTIC_ORIGIN_SUPPORTED

Priority 4
NON_ENDODONTIC_RED_FLAG

Priority 5
INDETERMINATE
```

This ordering is implemented programmatically.

The purpose is to preserve structural/fracture-related diagnostic pathways before conventional endodontic-origin classification or non-endodontic red-flag assignment.

---

# 19. Stage 7.2 — Gate 1

The first gate assesses suspected vertical root fracture.

The rule is:

```text
previously treated
AND
(
    isolated narrow deep pocket
    OR
    coronal/lateral sinus tract
    OR
    J-shaped / halo morphology
)
```

If satisfied:

```text
Category:
VRF_SUSPECTED

Priority:
1

Confidence:
indeterminate
```

---

# 20. Stage 7.2 — Gate 2

The second gate is evaluated only if the VRF gate is negative.

The rule is:

```text
(
    pain_on_release
    AND
    difficulty_localizing_pain
)
OR
visible_crack_line
```

If satisfied:

```text
Category:
CRACKED_TOOTH_SUSPECTED

Priority:
2
```

---

# 21. Stage 7.2 — Gate 3

The third gate is evaluated only after the fracture-related gates are negative.

The rule is:

```text
cold_response = False
AND
ept_response = False
```

If satisfied:

```text
Category:
ENDODONTIC_ORIGIN_SUPPORTED

Priority:
3

Confidence:
supported
```

---

# 22. Stage 7.2 — Gate 4

The fourth gate evaluates a non-endodontic red flag.

The rule is:

```text
(
    cold_response = True
    OR
    ept_response = True
)
AND
periapical lesion present
```

The resulting category is:

```text
NON_ENDODONTIC_RED_FLAG
```

This gate is only reached after the higher-priority fracture and standard endodontic-origin gates have been evaluated.

---

# 23. Stage 7.2 — Gate 5

If none of the previous gates are satisfied:

```text
INDETERMINATE
```

The system does not force a definitive classification when the available information is insufficient.

---

# 24. Module 13 — Authoritative Triage

The Stage 7.2 lesion-origin classification is merged with the existing red-flag assessment.

The resulting triage structure includes:

```text
category
priority
gate
confidence
basis
reason
```

The implementation preserves previous diagnostic states where necessary for auditability.

The Stage 7.2 classification is treated as authoritative for downstream reasoning.

---

# 25. Biological/Periapical Discordance

The architecture deliberately separates:

```text
Biological / clinical discordance
```

from:

```text
Lesion-origin classification
```

A periapical radiographic lesion does not automatically force the system into a conventional endodontic diagnosis.

Similarly, a suspected fracture case is not automatically relabeled as a non-endodontic case merely because a radiographic lesion is present.

This distinction is implemented programmatically.

---

# 26. Module 14 — Diagnostic Label Selection

The system subsequently selects probable pulpal and apical diagnostic labels.

The labels are based on the programmatic diagnostic-consistency state.

When the authoritative origin is:

```text
VRF_SUSPECTED
```

the apical diagnosis is set to:

```text
indeterminate
```

rather than forcing a conventional apical diagnosis.

This preserves the distinction between:

```text
Radiographic finding
```

and:

```text
Clinical diagnostic classification
```

---

# 27. Module 15 — Escalation Alignment

After diagnostic classification, the system aligns the escalation pathway.

Escalation can be influenced by:

* structural/fracture concerns;
* non-endodontic red flags;
* unresolved diagnostic uncertainty;
* target-specific findings;
* other programmatic safety conditions.

The LLM is not permitted to independently override the established escalation state.

---

# 28. Module 16 — Retrieval-Augmented Generation

The knowledge-retrieval component uses:

```text
BAAI/bge-small-en-v1.5
```

The default retrieval depth is:

```text
TOP_K = 5
```

The knowledge base contains:

```text
embeddings
metadata
processed text chunks
```

The system retrieves evidence dynamically according to the target-specific clinical context.

---

# 29. Dynamic Retrieval Query

The retrieval query is assembled from multiple information sources.

These include:

```text
Target tooth
Target-associated radiographic findings
Target-associated OPGAgent findings
Structured clinical findings
Diagnostic themes
```

The query can specifically emphasize concepts such as:

```text
Previous root canal treatment
Cracked tooth
Pain on release
Vertical root fracture
Symptomatic apical periodontitis
```

The retrieval logic also contains safeguards preventing a generic pain-on-biting pathway from overriding a higher-priority VRF retrieval pathway.

---

# 30. Module 17 — Evidence Retrieval

The BGE encoder generates a query embedding.

The system then compares this representation with the stored knowledge-base embeddings.

The retrieved result retains information including:

```text
Rank
Similarity score
Source
Pages
Chunk ID
Text
```

This evidence is passed to the reasoning layer.

---

# 31. Module 18 — Qwen2.5-VL Reasoning

The final reasoning component uses:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

The model receives a structured context containing:

```text
Target tooth
Radiographic findings
OPGAgent findings
Clinical history
Structured clinical findings
Diagnostic consistency
Lesion-origin classification
Triage
Escalation
Retrieved evidence
Uncertainties
```

The LLM is explicitly instructed that programmatic decisions are authoritative.

---

# 32. LLM Role Boundaries

The v65 architecture explicitly limits the role of the LLM.

The LLM:

### Can

* ask focused clinical questions;
* summarize clinical information;
* synthesize radiographic observations;
* integrate retrieved evidence;
* explain the reasoning;
* identify missing information.

### Cannot override

* programmatic diagnostic labels;
* deterministic triage;
* red-flag classification;
* Stage 7.2 lesion-origin classification;
* escalation decisions;
* target-tooth lock.

This is a central safety principle of the architecture.

---

# 33. Module 19 — Diagnostic Reasoning Prompt

The final reasoning prompt contains explicit constraints.

The model is instructed to:

* remain target-tooth locked;
* use only the supplied visual findings;
* use structured clinical findings;
* respect the authoritative diagnostic state;
* use retrieved evidence;
* preserve uncertainty;
* avoid unsupported rare diagnoses;
* distinguish PAI from clinical diagnosis;
* follow the established escalation pathway.

The prompt therefore acts as an additional control layer around the language model.

---

# 34. Module 20 — Safety Validation

The generated response is passed through programmatic validation.

The validator checks for inappropriate statements such as:

```text
confirms pulp vitality
proves pulp vitality
indicates a vital pulp
```

when the clinical context contains a positive cold response.

It also checks for inappropriate use of radiographic indices as clinical diagnoses.

The architecture therefore contains:

```text
LLM generation
      ↓
Safety validation
      ↓
Accepted / flagged output
```

rather than returning the raw LLM response directly.

---

# 35. Module 21 — Structured Audit Output

The final system constructs a structured JSON record.

The output contains information corresponding to:

```text
System metadata
Radiographic findings
OPGAgent findings
Clinical history
Structured clinical findings
Diagnostic consistency
Probable diagnosis
Non-endodontic assessment
Triage
Lesion-origin classification
Escalation
Reasoning
Raw model response
Safety validation
Uncertainties
Evidence references
Specialist verification requirement
```

This allows the complete diagnostic pathway to be examined after inference.

---

# 36. Audit Trail

The architecture is intentionally traceable.

A case can be followed through:

```text
INPUT
  │
  ▼
VISION
  │
  ▼
OPGAGENT
  │
  ▼
TARGET TOOTH
  │
  ▼
CLINICAL HISTORY
  │
  ▼
STRUCTURED FINDINGS
  │
  ▼
DIAGNOSTIC CONSISTENCY
  │
  ▼
STAGE 7.2
  │
  ▼
TRIAGE
  │
  ▼
RETRIEVAL
  │
  ▼
LLM REASONING
  │
  ▼
SAFETY VALIDATION
  │
  ▼
STRUCTURED OUTPUT
```

This is intended to facilitate error analysis and research reproducibility.

---

# 37. Model Loading Architecture

The v65 system loads different components from different sources.

Conceptually:

```text
Hugging Face
    │
    ├── Qwen2.5-VL-7B-Instruct
    │
    └── BAAI/bge-small-en-v1.5

ENDOLLM project
    │
    ├── Frozen vision pipeline
    ├── Diagnostic rules
    └── Red-flag rules

OPGAgent
    │
    ├── Tooth enumeration
    └── Visual expert models

Knowledge base
    │
    ├── Embeddings
    ├── Metadata
    └── Text chunks
```

The exact redistribution status of external models and data should be checked before including those files in the public repository.

---

# 38. Runtime Architecture

The v65 script uses:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

with 4-bit quantization.

The configuration includes:

```text
NF4
Double quantization
Float16 computation
Automatic device mapping
```

Generation uses:

```text
do_sample = False
repetition_penalty = 1.05
```

The image-processing bounds are:

```text
MIN_PIXELS = 256 × 28 × 28
MAX_PIXELS = 2048 × 28 × 28
```

These are runtime inference parameters.

---

# 39. Retrieval Architecture

The retrieval architecture can be represented as:

```text
Target-specific clinical/radiographic context
                    │
                    ▼
             Retrieval query
                    │
                    ▼
       BGE-small-en-v1.5 encoder
                    │
                    ▼
             Query embedding
                    │
                    ▼
         Similarity comparison
                    │
                    ▼
               Top-K = 5
                    │
                    ▼
          Evidence chunks
                    │
                    ▼
          Qwen reasoning layer
```

The retrieved evidence is retained in the structured output for auditability.

---

# 40. Clinical Safety Architecture

Safety is distributed across several layers.

```text
Layer 1
Target-tooth confirmation

Layer 2
Structured clinical extraction

Layer 3
Deterministic diagnostic rules

Layer 4
Stage 7.2 lesion-origin precedence

Layer 5
Authoritative triage

Layer 6
LLM prompt constraints

Layer 7
Programmatic response validation

Layer 8
Specialist verification requirement
```

This layered design reduces dependence on any single model component.

---

# 41. Failure Containment

The architecture is designed so that an error in one component does not automatically become an unrestricted diagnostic conclusion.

For example:

```text
Visual finding
      ↓
must be target-associated
      ↓
must be interpreted with clinical findings
      ↓
must pass deterministic diagnostic hierarchy
      ↓
must respect triage
      ↓
LLM explains rather than overrides
      ↓
output is validated
```

This provides a structured pathway for containing errors.

---

# 42. Uncertainty Architecture

Uncertainty is explicitly preserved.

The system can represent:

```text
None / Unknown
Indeterminate confidence
INDETERMINATE category
Missing information
Specialist verification required
Escalation required
```

This prevents incomplete clinical information from automatically producing a definitive diagnosis.

---

# 43. Output Architecture

The final structured output is written to:

```text
phase2_knowledge/evaluation/phase4_outputs/
```

using the pattern:

```text
{image_stem}_phase4_structured_diagnostic.json
```

The JSON output is intended to provide a machine-readable audit record.

---

# 44. End-to-End Architecture

The complete v65 architecture can therefore be summarized as:

```text
                         ┌─────────────────────┐
                         │     RADIOGRAPH      │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │ Frozen Vision     │               │ OPGAgent          │
        │                   │               │                   │
        │ Faster R-CNN      │               │ Tooth enumeration │
        │ ResNet-50         │               │ Visual experts    │
        └─────────┬─────────┘               └─────────┬─────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                         ┌────────────────────┐
                         │ Target-Tooth Gate  │
                         │ Clinician confirms │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         Target-specific evidence
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ Clinical History   │
                         │ Qwen ≤5 questions  │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         Structured clinical data
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐              ┌────────────────────┐
        │ Diagnostic        │              │ Stage 7.2          │
        │ consistency       │              │ lesion-origin       │
        │ rules             │              │ classifier          │
        └─────────┬─────────┘              └──────────┬─────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                             Authoritative triage
                                    │
                                    ▼
                           Dynamic RAG query
                                    │
                                    ▼
                           BGE retrieval
                                    │
                                    ▼
                            Evidence chunks
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ Qwen2.5-VL-7B     │
                         │ evidence-grounded │
                         │ reasoning          │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         Programmatic validation
                                    │
                                    ▼
                         Structured JSON audit
```

---

# 45. Architecture Design Principles

The v65 implementation is based on the following principles:

### 1. Target-tooth grounding

The system requires explicit clinician identification of the symptomatic tooth.

### 2. Separation of perception and reasoning

Visual models generate observations; the downstream system determines how those observations are used.

### 3. Deterministic clinical precedence

Higher-priority structural and diagnostic pathways are established programmatically.

### 4. LLM constraint

The LLM synthesizes and explains rather than overriding the authoritative clinical state.

### 5. Evidence retrieval

Relevant knowledge is retrieved dynamically rather than relying solely on the pretrained LLM.

### 6. Explicit uncertainty

Unknown information remains unknown.

### 7. Safety validation

Generated text is checked before final output.

### 8. Auditability

Intermediate states are retained in structured form.

---

# 46. Architecture Limitations

The architecture remains dependent on several external and upstream components.

These include:

* frozen radiographic models;
* OPGAgent;
* pretrained Qwen2.5-VL;
* BGE retrieval encoder;
* knowledge-base quality;
* clinician-provided clinical information.

Consequently, an error at an upstream stage may propagate downstream.

The architecture reduces this risk through target locking, deterministic rules, precedence logic, retrieval, and safety validation, but does not eliminate it.

---

# 47. Repository Mapping

The repository documentation corresponds to the architecture as follows:

```text
README.md
    ↓
Project overview

DATASET.md
    ↓
Data and annotations

TRAINING.md
    ↓
Model-development status

EVALUATION.md
    ↓
Evaluation framework

ARCHITECTURE.md
    ↓
Technical system architecture

MANUSCRIPT_REPRODUCIBILITY.md
    ↓
Manuscript ↔ code ↔ data mapping
```

The current v65 implementation remains the principal executable integration artifact.

---

# 48. Current Version

```text
ENDOLLM
Phase 4.5 Interactive Endodontic MLLM
Version: v65

Overall project stage:
7/9

Diagnostic-rule stage:
7.2
```

This architecture document describes the implementation represented by the v65 code.

Future architectural changes should be versioned and documented explicitly.

---

# 49. Summary

ENDOLLM v65 is a **hybrid, clinician-supervised multimodal endodontic decision-support architecture**.

Its defining feature is the separation of:

```text
Perception
    ↓
Target-tooth grounding
    ↓
Clinical information
    ↓
Deterministic diagnostic logic
    ↓
Evidence retrieval
    ↓
LLM synthesis
    ↓
Safety validation
    ↓
Audit output
```

The architecture therefore does not rely on an unrestricted LLM to independently determine the diagnosis.

Instead, the LLM operates within a structured computational and clinical framework in which target selection, diagnostic precedence, triage, escalation, and safety constraints are programmatically represented.
