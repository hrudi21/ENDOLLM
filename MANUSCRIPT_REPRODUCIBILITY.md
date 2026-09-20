# ENDOLLM Manuscript Reproducibility Guide

## Manuscript–Code–Data Reproducibility Mapping

**Project:** ENDOLLM
**Current implementation:** Phase 4.5 Interactive Endodontic MLLM (v65)
**Overall project stage:** 7/9
**Current diagnostic-rule stage:** 7.2

---

# 1. Purpose

This document provides a reproducibility map connecting the ENDOLLM research manuscript with the corresponding:

* source code;
* dataset;
* model components;
* diagnostic rules;
* retrieval system;
* evaluation procedures;
* structured outputs;
* version information.

The purpose is to allow reviewers and researchers to understand which repository components correspond to the computational methods described in the manuscript.

The repository is designed to distinguish between:

```text id="c3h9ri"
Original/frozen models
        ↓
External pretrained models
        ↓
ENDOLLM programmatic components
        ↓
Integrated v65 inference pipeline
        ↓
Structured outputs
```

---

# 2. Manuscript Association

The GitHub repository is intended to provide the computational implementation associated with the ENDOLLM research manuscript.

The manuscript title, journal-specific metadata, DOI, and final publication information should be inserted here once the manuscript version and bibliographic information are finalized.

```text
Manuscript title:
[INSERT FINAL MANUSCRIPT TITLE]

Target journal:
Computers in Biology and Medicine

Manuscript status:
[INSERT STATUS]

DOI:
[INSERT WHEN AVAILABLE]

Repository:
[INSERT FINAL GITHUB REPOSITORY URL]
```

These fields are intentionally left open rather than populated with unverified information.

---

# 3. Version Correspondence

The manuscript-associated implementation should be identified explicitly.

Current repository version:

```text id="vrsj86"
ENDOLLM
Phase 4.5 Interactive Endodontic MLLM
v65
Stage 7.2
```

The principal implementation file is:

```text id="d3b1t2"
phase4_interactive_endodontic_mllm_v65.py
```

The repository should be version-tagged once the manuscript-associated computational snapshot is finalized.

---

# 4. Main Reproducibility Artifact

The principal executable implementation is:

```text id="2i5c4a"
phase4_interactive_endodontic_mllm_v65.py
```

This script integrates the major components of the ENDOLLM system.

Its workflow includes:

1. radiographic analysis;
2. OPGAgent visual analysis;
3. target-tooth identification;
4. clinical history elicitation;
5. clinical normalization;
6. structured clinical extraction;
7. diagnostic consistency assessment;
8. red-flag assessment;
9. Stage 7.2 lesion-origin classification;
10. triage merging;
11. diagnostic label selection;
12. escalation alignment;
13. dynamic retrieval;
14. evidence-grounded LLM reasoning;
15. safety validation;
16. structured JSON output.

---

# 5. Repository Documentation Map

The repository currently contains the following principal documentation:

```text id="c3c18v"
README.md
DATASET.md
TRAINING.md
EVALUATION.md
ARCHITECTURE.md
MANUSCRIPT_REPRODUCIBILITY.md
```

Their purposes are:

| File                            | Purpose                                      |
| ------------------------------- | -------------------------------------------- |
| `README.md`                     | Overall system description                   |
| `DATASET.md`                    | Dataset composition and provenance framework |
| `TRAINING.md`                   | Training/model-development status            |
| `EVALUATION.md`                 | Evaluation framework                         |
| `ARCHITECTURE.md`               | Detailed technical architecture              |
| `MANUSCRIPT_REPRODUCIBILITY.md` | Manuscript-to-repository mapping             |

Additional repository files will be added during the reproducibility release process.

---

# 6. Manuscript Methods ↔ Repository Mapping

The following mapping can be used to connect manuscript methods to the corresponding computational components.

| Manuscript component           | Repository component                                                   |
| ------------------------------ | ---------------------------------------------------------------------- |
| Radiographic analysis          | `phase4_interactive_endodontic_mllm_v65.py` + frozen `vision_pipeline` |
| Radiographic detector          | `FasterRCNN_epoch14`                                                   |
| Radiographic classifier        | `ResNet50_epoch12`                                                     |
| Tooth enumeration              | OPGAgent YOLO component                                                |
| Additional visual analysis     | OPGAgent TVEM/MaskDINO components                                      |
| Target-tooth identification    | Interactive target-tooth gate                                          |
| Clinical history acquisition   | Qwen2.5-VL interaction                                                 |
| Clinical normalization         | v65 clinical normalization functions                                   |
| Structured clinical extraction | v65 structured extraction functions                                    |
| Diagnostic consistency         | `diagnostic_rules.py`                                                  |
| Non-endodontic red flags       | `non_endodontic_red_flags.py`                                          |
| Lesion-origin classification   | Stage 7.2 deterministic classifier                                     |
| Evidence retrieval             | BGE-small retrieval pipeline                                           |
| Knowledge base                 | `phase2_knowledge/`                                                    |
| LLM reasoning                  | Qwen2.5-VL-7B-Instruct                                                 |
| Safety validation              | v65 response validation                                                |
| Structured output              | `phase4_outputs/` JSON files                                           |

---

# 7. Radiographic Vision Component

The v65 implementation uses an established/frozen radiographic vision pipeline.

The models recorded by the current implementation are:

```text id="w0u8p8"
FasterRCNN_epoch14
ResNet50_epoch12
```

The terminology framework recorded in the structured output is:

```text id="smr2si"
2025 AAE/ESE periapical terminology
```

The v65 implementation calls the existing radiographic analysis pipeline rather than retraining the models during execution.

Therefore, reproducibility of the radiographic component requires access to the corresponding frozen model implementation and checkpoints, subject to their redistribution permissions.

---

# 8. OPGAgent Component

The v65 implementation adds OPGAgent visual evidence.

The current implementation references:

```text id="qz4v5f"
OPGAgent/
```

including:

```text id="pfj4q3"
api_service/yolo_enumeration/model/best.pt
```

and additional visual-expert checkpoints.

The OPGAgent component contributes:

* tooth enumeration;
* quadrant information;
* visual findings;
* bone-loss information;
* tooth-associated findings.

The exact external OPGAgent version and redistribution instructions should be documented once the final repository dependency structure is frozen.

---

# 9. Target-Tooth Grounding

The manuscript's description of target-specific reasoning should correspond to the interactive target-tooth gate in v65.

The computational sequence is:

```text id="n9x7m2"
Radiographic findings
        ↓
OPGAgent findings
        ↓
Clinician identifies symptomatic tooth/teeth
        ↓
System validates tooth identity
        ↓
Clinician confirms
        ↓
Target-specific evidence
```

Only findings associated with the confirmed target tooth are intended to proceed as target-specific diagnostic evidence.

This is an important reproducibility feature and should be explicitly described in the manuscript methods.

---

# 10. Clinical History Acquisition

The v65 implementation uses Qwen2.5-VL to generate focused clinical questions.

The system permits a maximum of:

```text id="7v7j2r"
5 questions
```

The questions address clinically relevant features such as:

* pain characteristics;
* thermal response;
* EPT;
* percussion;
* palpation;
* biting/release pain;
* periodontal findings;
* crack-related findings;
* previous treatment;
* trauma;
* swelling and sinus tract.

The model is instructed to gather information rather than directly diagnose during this stage.

---

# 11. Structured Clinical Representation

The free-text clinical history is converted into a structured representation.

The implementation uses three states:

```text id="z8kqvl"
True
False
None / Unknown
```

The distinction between false and unknown is important for reproducibility.

Missing information is not automatically converted to a negative finding.

The structured representation includes variables covering:

```text id="p4z7p2"
Symptoms
Thermal testing
EPT
Percussion
Palpation
Biting/release pain
Pain localization
Swelling
Sinus tract
Periodontal findings
Crack findings
Previous treatment
Lesion morphology
Trauma
Systemic infection signs
```

---

# 12. Diagnostic Rule Implementation

The manuscript's description of deterministic clinical reasoning should correspond to the programmatic components rather than to free-form LLM interpretation.

The v65 implementation imports:

```text id="7s0s34"
evaluate_diagnostic_consistency
assess_non_endodontic_red_flags
```

from the ENDOLLM project components.

The resulting programmatic diagnostic state is then passed into the downstream reasoning process.

---

# 13. Stage 7.2 Lesion-Origin Classification

The current v65 implementation contains an explicit lesion-origin hierarchy.

```text id="16s2a8"
1. VRF suspected
2. Cracked tooth suspected
3. Standard endodontic origin supported
4. Non-endodontic red flag
5. Indeterminate
```

The corresponding categories are:

```text id="0o0c7v"
VRF_SUSPECTED
CRACKED_TOOTH_SUSPECTED
ENDODONTIC_ORIGIN_SUPPORTED
NON_ENDODONTIC_RED_FLAG
INDETERMINATE
```

This classification is deterministic and is not generated solely by Qwen.

---

# 14. Stage 7.2 Reproducibility

The Stage 7.2 rules should be reproducible from the source code.

### Gate 1 — VRF

```text id="3j0m57"
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

### Gate 2 — Cracked tooth

```text id="2k4f0a"
(
    pain on release
    AND
    difficulty localizing pain
)
OR
visible crack line
```

### Gate 3 — Standard endodontic origin

```text id="3b5s8p"
cold response = False
AND
EPT response = False
```

### Gate 4 — Non-endodontic red flag

```text id="1m0t50"
(
    cold response = True
    OR
    EPT response = True
)
AND
periapical lesion present
```

### Gate 5 — Indeterminate

If no preceding gate is satisfied:

```text id="8j9d9w"
INDETERMINATE
```

These rules should be kept synchronized with the manuscript's description.

---

# 15. Triage and Escalation

The v65 implementation preserves the distinction between:

```text id="d2e4d4"
Diagnostic classification
Triage
Escalation
```

The Stage 7.2 classifier provides an authoritative lesion-origin state.

The downstream system then aligns the escalation pathway with that state.

This means that the final LLM narrative should not be interpreted independently of the structured triage and escalation fields.

---

# 16. Retrieval-Augmented Generation

The manuscript's RAG methodology corresponds to the following v65 components.

### Retrieval encoder

```text id="d0a11c"
BAAI/bge-small-en-v1.5
```

### Default retrieval depth

```text id="b8w7x7"
TOP_K = 5
```

### Knowledge-base resources

```text id="y0h2vb"
bge_small_embeddings.npy
bge_small_metadata.jsonl
chunks.jsonl
```

and merged resources:

```text id="i4t3tw"
bge_small_embeddings_merged_244.npy
bge_small_metadata_merged_244.jsonl
chunks_merged_244.jsonl
```

The retrieval query incorporates target-specific clinical and visual information.

---

# 17. Language Model

The manuscript's multimodal language-model component corresponds to:

```text id="4cc5k0"
Qwen/Qwen2.5-VL-7B-Instruct
```

The v65 implementation uses the pretrained model for inference.

It does not perform Qwen fine-tuning inside the v65 script.

The model is loaded using 4-bit quantization with:

```text id="i1m4jb"
NF4
Double quantization
Float16 computation
Automatic device mapping
```

---

# 18. LLM Role in the Architecture

The manuscript should distinguish between:

```text id="1mt0yt"
LLM reasoning
```

and:

```text id="0d4j35"
programmatic diagnostic decision
```

The LLM is used for:

* interactive questioning;
* clinical-language processing;
* evidence synthesis;
* explanation.

The programmatic components establish:

* diagnostic consistency;
* lesion-origin classification;
* triage;
* red flags;
* escalation.

This distinction is central to the architecture.

---

# 19. Safety Validation

The v65 implementation validates generated output before finalization.

The validation layer checks for inappropriate statements including unsupported claims concerning pulp vitality and inappropriate interpretation of PAI.

This corresponds to a manuscript methodology in which generated reasoning is not accepted without programmatic validation.

---

# 20. Structured Output

The v65 system generates a structured JSON audit record.

The output contains fields representing:

```text id="o5a6y5"
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
Specialist verification
```

The output directory is:

```text id="xwz2py"
phase2_knowledge/evaluation/phase4_outputs/
```

with filenames following:

```text id="1wyb8r"
{image_stem}_phase4_structured_diagnostic.json
```

---

# 21. Reproducing a Single Case

A reviewer with the required dependencies can conceptually reproduce an individual case using:

```text id="4s9q6v"
Input radiograph
        ↓
Frozen visual analysis
        ↓
OPGAgent
        ↓
Target-tooth selection
        ↓
Clinical history
        ↓
Structured clinical extraction
        ↓
Stage 7.2 rules
        ↓
Dynamic retrieval
        ↓
Qwen reasoning
        ↓
Safety validation
        ↓
Structured JSON
```

The exact execution environment and external model locations must be configured according to the repository's final dependency instructions.

---

# 22. Manuscript Results ↔ Evaluation

The manuscript's reported numerical results should be linked to the corresponding evaluation artifacts.

The final repository should provide, where permitted:

```text id="4xq8p6"
Dataset version
Evaluation script
Model/checkpoint version
Evaluation output
Statistical analysis
Tables
Figures
```

The current v65 script is primarily the integrated inference implementation and should not be presented as the complete statistical analysis pipeline.

---

# 23. Numerical Results Policy

Numerical results reported in the manuscript should be traceable to:

1. a defined dataset split;
2. a defined ENDOLLM version;
3. a defined model/checkpoint;
4. a defined evaluation procedure;
5. a defined statistical analysis.

Results should not be reproduced from memory or manually re-entered into repository documentation without an identifiable analysis source.

---

# 24. Dataset Reproducibility

The radiographic dataset is documented in:

```text id="mnyb4k"
DATASET.md
```

The documented dataset contains:

```text id="v3v3js"
3,924 radiographs
11,772 JSONL examples

Training:
2,747 radiographs

Validation:
588 radiographs

Test:
589 radiographs
```

The dataset includes:

* classification examples;
* counting examples;
* localization examples;
* periapical-lesion bounding boxes;
* PAI classes 3–5.

Dataset redistribution should be governed by the original dataset's licensing and permission conditions.

---

# 25. Model Reproducibility

The repository should identify every external model by:

```text id="zqgh23"
Model name
Version/checkpoint
Source
License
Download method
Expected local path
```

The v65 code references external checkpoints and project directories.

These should be converted to documented, configurable dependencies before the final public reproducibility release.

---

# 26. Environment Reproducibility

The current v65 code uses machine-relative paths such as:

```text id="s3qk79"
~/EndoMLLM/
~/OPGAgent/
~/opgagent_env/
```

For public release, these paths should ideally be replaced by configurable paths or environment variables.

The final repository should document:

```text id="j0i6q7"
Python version
PyTorch version
Transformers version
Ultralytics version
SentenceTransformers version
BitsAndBytes version
CUDA/device requirements
Other dependencies
```

Only verified versions should be recorded.

---

# 27. Reproducibility and External Dependencies

ENDOLLM depends on components that may be distributed separately.

These include:

* Qwen2.5-VL-7B-Instruct;
* BAAI/bge-small-en-v1.5;
* OPGAgent;
* OPGAgent model checkpoints;
* frozen ENDOLLM vision models;
* radiographic datasets;
* RAG knowledge-base resources.

The repository should provide references/download instructions rather than redistributing restricted resources.

---

# 28. Version Control and Manuscript Snapshot

Once the manuscript-associated implementation is finalized, a version tag should be created.

Recommended concept:

```text
v65
```

or a more explicit release identifier such as:

```text
ENDOLLM-v65
```

The exact release tag should be synchronized with the manuscript's computational version.

Future changes should use new commits/releases rather than modifying the manuscript-associated implementation without version history.

---

# 29. Recommended Reviewer Workflow

A reviewer should be able to follow this sequence:

```text id="84j8b8"
1. Read README.md
        ↓
2. Read ARCHITECTURE.md
        ↓
3. Read DATASET.md
        ↓
4. Read TRAINING.md
        ↓
5. Read EVALUATION.md
        ↓
6. Read this reproducibility guide
        ↓
7. Inspect v65 source code
        ↓
8. Obtain permitted external dependencies
        ↓
9. Reconstruct the runtime environment
        ↓
10. Execute permitted evaluation cases
        ↓
11. Inspect structured JSON outputs
```

This provides a transparent pathway from manuscript description to computational implementation.

---

# 30. What the Repository Does Not Claim

The repository does not claim that:

* v65 itself retrains Qwen2.5-VL;
* v65 retrains the frozen Faster R-CNN/ResNet-50 models;
* every external model checkpoint is redistributed;
* every original dataset image is publicly redistributed;
* every statistical analysis is contained in the v65 script;
* the LLM independently establishes the final diagnosis.

These distinctions are important for accurate interpretation of the research system.

---

# 31. Clinical Reproducibility

Reproduction of the complete clinical workflow requires more than running a Python script.

The workflow also includes clinician interaction.

In particular:

```text id="3hby9r"
Target-tooth identification
Clinical history
Clinical examination findings
Clinical testing information
```

are incorporated into the diagnostic process.

Therefore, computational reproducibility should be distinguished from clinical reproducibility.

A software-only execution without equivalent clinical inputs does not reproduce the complete clinical decision-support scenario.

---

# 32. Audit Reproducibility

The structured output allows investigators to inspect intermediate decisions.

A reproducibility analysis should therefore compare not only the final narrative but also:

```text id="l8xq5q"
Target tooth
Structured clinical findings
Diagnostic consistency
Lesion-origin classification
Triage
Retrieved evidence
Escalation
Safety validation
```

This allows differences between two executions to be localized to a specific stage.

---

# 33. Recommended Release Contents

The final manuscript-associated release should contain, where legally and technically possible:

```text id="c7s7hj"
README.md
DATASET.md
TRAINING.md
EVALUATION.md
ARCHITECTURE.md
MANUSCRIPT_REPRODUCIBILITY.md
phase4_interactive_endodontic_mllm_v65.py
```

Additional files should include:

```text id="1m6a5d"
Model card
Citation file
License
Environment specification
Git ignore configuration
Environment-variable template
Evaluation scripts
```

when finalized.

---

# 34. Final Reproducibility Checklist

Before submitting the repository link to the journal, verify:

### Code

* [ ] v65 source is uploaded.
* [ ] Code corresponds to the manuscript-associated version.
* [ ] No passwords or API keys are present.
* [ ] No machine-specific private files are included.
* [ ] External paths are documented.

### Data

* [ ] Dataset source is identified.
* [ ] Dataset license is verified.
* [ ] Train/validation/test split is documented.
* [ ] Redistribution permissions are verified.

### Models

* [ ] Model names are documented.
* [ ] Checkpoint versions are documented.
* [ ] Model licenses are checked.
* [ ] Download instructions are provided where necessary.

### RAG

* [ ] Knowledge-base source is documented.
* [ ] Retrieval encoder is identified.
* [ ] Retrieval index version is documented.
* [ ] Knowledge-base redistribution status is checked.

### Evaluation

* [ ] Test set is identified.
* [ ] Evaluation procedure is documented.
* [ ] Statistical analysis is documented.
* [ ] Manuscript results can be traced to an analysis artifact.

### Versioning

* [ ] Manuscript-associated release is tagged.
* [ ] Repository commit corresponds to the manuscript version.
* [ ] Major future changes are versioned separately.

---

# 35. Final Manuscript–Repository Mapping

The intended reproducibility relationship is:

```text id="m0g6gp"
                         MANUSCRIPT
                             │
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
          METHODS         RESULTS        DISCUSSION
             │               │
             ▼               ▼
       Repository code    Evaluation
             │               │
             ├───────────────┤
             │
             ▼
       ENDOLLM v65
             │
     ┌───────┼────────┐
     │       │        │
     ▼       ▼        ▼
   DATA    MODELS   RULES
     │       │        │
     └───────┼────────┘
             ▼
        RAG + LLM
             │
             ▼
       Structured output
```

The objective is that a reviewer can move from any major computational claim in the manuscript to the corresponding documented implementation component.

---

# 36. Current Status

```text id="k5u5n5"
ENDOLLM project stage: 7/9

Current implementation:
Phase 4.5 Interactive Endodontic MLLM

Version:
v65

Diagnostic-rule stage:
7.2

Repository status:
Manuscript-associated reproducibility release in preparation
```

---

# 37. Important Final Note

This repository should be considered a **research reproducibility resource**, not a claim of autonomous clinical validation.

The system is designed for clinician-supervised endodontic diagnostic decision support.

Clinical interpretation remains dependent on appropriate professional examination, testing, and verification.

---

## Summary

The purpose of this document is to establish a transparent relationship between:

```text
Manuscript
   ↕
ENDOLLM v65 code
   ↕
Dataset
   ↕
Models
   ↕
Diagnostic rules
   ↕
RAG knowledge base
   ↕
Evaluation
   ↕
Structured outputs
```

The final repository should preserve this relationship through version control, documented dependencies, reproducible evaluation procedures, and explicit identification of external components.
