# ENDOLLM Evaluation Documentation

## Overview

This document describes the evaluation framework for the **ENDOLLM** multimodal endodontic AI system and clarifies the distinction between:

1. evaluation of individual visual components;
2. evaluation of the integrated ENDOLLM pipeline;
3. evaluation of diagnostic reasoning;
4. evaluation of safety and triage behavior; and
5. evaluation of reproducibility and auditability.

The current implementation is:

```text
ENDOLLM
Phase 4.5 Interactive Endodontic MLLM
Version: v65
Project stage: 7/9
Diagnostic-rule stage: 7.2
```

The v65 implementation is an inference/integration system. Therefore, evaluation should be performed at both the **component level** and the **complete-system level**.

---

# 1. Evaluation Philosophy

ENDOLLM is not a single neural-network classifier.

It is a hybrid system comprising:

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
Multimodal LLM reasoning
        +
Safety validation
```

Consequently, a single performance metric cannot completely characterize the system.

Evaluation should consider:

* visual detection;
* visual classification;
* lesion localization;
* target-tooth grounding;
* clinical diagnostic consistency;
* lesion-origin classification;
* triage/red-flag identification;
* evidence retrieval;
* LLM reasoning;
* safety behavior;
* final diagnostic agreement;
* structured-output integrity.

---

# 2. Dataset Partitions

The manuscript-associated radiographic dataset is documented separately in:

```text
DATASET.md
```

The verified dataset contains:

```text
Total radiographs:       3,924
Total JSONL examples:   11,772

Training:
    2,747 radiographs
    8,241 examples

Validation:
      588 radiographs
    1,764 examples

Test:
      589 radiographs
    1,767 examples
```

Each image contributes three task examples:

```text
classification
counting
localization
```

The test partition should remain independent of model development and parameter optimization.

---

# 3. Component-Level Evaluation

## 3.1 Radiographic Detection

The frozen radiographic detection component is identified in v65 as:

```text
FasterRCNN_epoch14
```

Its role is radiographic lesion detection.

Evaluation of this component should assess the correspondence between predicted lesion locations and the reference annotations.

Where appropriate, localization performance may be described using metrics appropriate to object detection and lesion localization.

The exact metric set should correspond to the original model-development protocol and should not be inferred solely from the v65 inference script.

---

# 4. Radiographic Classification

The frozen radiographic classification component is identified as:

```text
ResNet50_epoch12
```

The model is used by the frozen vision pipeline and is not retrained by v65.

Classification evaluation should use the designated test partition and the reference labels associated with the radiographic dataset.

The exact classification metrics should be reported according to the original experimental protocol.

Possible reporting fields include:

```text
Accuracy
Sensitivity
Specificity
Precision
F1 score
Confusion matrix
```

Only metrics actually calculated in the corresponding experiment should be reported in the manuscript.

---

# 5. Lesion Localization

The localization task contains:

* periapical-lesion bounding boxes;
* PAI classes 3–5;
* original-image pixel coordinates.

Localization evaluation should compare predicted lesion locations with the corresponding reference annotations.

The exact threshold and metric used for localization should be reported in the associated experiment documentation.

The v65 integration code itself does not define the original model-training evaluation protocol.

---

# 6. Lesion Counting

The dataset includes a dedicated lesion-counting task.

Evaluation should compare:

```text
Predicted lesion count
vs.
Reference lesion count
```

The exact statistical metric should be specified in the corresponding experiment.

Possible approaches include:

* exact count agreement;
* mean absolute error;
* count-distribution analysis.

The final manuscript should report only the metric actually used in the experiment.

---

# 7. OPGAgent Evaluation

The v65 system incorporates OPGAgent as an additional visual-evidence layer.

Its components include:

```text
Tooth enumeration
Quadrant identification
Disease/finding detection
Mandibular/maxillary or sinus-related findings
Bone-loss assessment
```

The current v65 configuration uses:

```text
Tooth enumeration confidence threshold: 0.25
TVEM confidence threshold:              0.30
```

These thresholds are inference parameters.

OPGAgent performance should be evaluated separately from the ENDOLLM reasoning layer because errors in tooth enumeration or visual finding detection can propagate into downstream target-tooth reasoning.

---

# 8. Target-Tooth Grounding Evaluation

A defining feature of v65 is the **interactive target-tooth gate**.

The system does not automatically assume that every radiographic abnormality belongs to the symptomatic tooth.

Instead:

```text
Radiographic findings
        ↓
OPGAgent tooth-level findings
        ↓
Clinician identifies symptomatic tooth
        ↓
System validates target tooth
        ↓
Target-specific evidence
```

Target-tooth evaluation should therefore assess whether the final reasoning remains restricted to the clinician-confirmed target tooth.

Relevant evaluation questions include:

1. Was the target tooth correctly identified?
2. Were unrelated findings excluded from target-specific evidence?
3. Were target-associated findings retained?
4. Did the final reasoning refer to the correct tooth?
5. Did the system incorrectly attribute an unrelated lesion to the target tooth?

---

# 9. Target-Specific Evidence Association

The v65 implementation uses spatial association between detected findings and teeth.

The configured association criteria include:

```text
Minimum IoU:                0.05
Minimum lesion containment: 0.50
```

Evaluation of this component should examine whether:

```text
Detected finding
        ↓
Correct tooth association
```

is maintained.

This is important because target-tooth grounding is a prerequisite for meaningful downstream diagnostic reasoning.

---

# 10. Clinical History Evaluation

The v65 system uses Qwen2.5-VL to generate a limited set of focused clinical questions.

The maximum number of questions is:

```text
5
```

The questions are intended to identify clinically relevant missing information without directly providing a diagnosis.

Evaluation may therefore assess:

* relevance of questions;
* coverage of important clinical features;
* avoidance of premature diagnosis;
* identification of missing information;
* appropriateness to the selected target tooth.

The current v65 source does not define a validated numerical scoring system for question quality.

Such a metric should therefore be added only if formally developed and evaluated.

---

# 11. Structured Clinical Extraction

Clinical history is converted into a structured representation.

The system uses three-state logic:

```text
TRUE
FALSE
UNKNOWN / NONE
```

This is important for evaluation because:

> Unknown clinical information should not be treated as a negative finding.

Evaluation of clinical extraction can compare the structured representation against clinician-entered reference information.

Potential measures include:

* field-level agreement;
* sensitivity for positive findings;
* specificity for negative findings;
* unknown-state agreement.

The exact metric should be specified if such a validation experiment is performed.

---

# 12. Diagnostic Consistency Evaluation

ENDOLLM performs programmatic diagnostic consistency assessment before final LLM reasoning.

The system incorporates:

```text
evaluate_diagnostic_consistency(...)
```

The evaluation should determine whether the programmatic classification is consistent with the reference clinical findings.

This component should be evaluated separately from the language model because the classification is deterministic.

---

# 13. Stage 7.2 Lesion-Origin Evaluation

The v65 implementation contains a deterministic lesion-origin hierarchy:

```text
Priority 1 — VRF suspected
Priority 2 — Cracked tooth suspected
Priority 3 — Standard endodontic origin supported
Priority 4 — Non-endodontic red flag
Priority 5 — Indeterminate
```

The hierarchy should be evaluated against an appropriate clinician/reference-standard classification.

Possible evaluation outputs include:

```text
Overall agreement
Category-level agreement
Confusion matrix
False-positive fracture classification
False-negative fracture classification
Indeterminate classification rate
```

The exact reference standard must be defined before reporting these metrics.

---

# 14. Vertical Root Fracture Pathway

The first Stage 7.2 gate concerns suspected vertical root fracture.

The implemented rule requires:

```text
Previously treated
AND
(
    isolated narrow deep pocket
    OR
    coronal/lateral sinus tract
    OR
    J-shaped / halo morphology
)
```

Evaluation should determine whether this rule correctly identifies the intended clinical cases.

Because this is a deterministic rule, performance can be reported separately from LLM performance.

---

# 15. Cracked-Tooth Pathway

The second Stage 7.2 gate concerns suspected cracked tooth.

The rule includes:

```text
(
    pain on release
    AND
    difficulty localizing pain
)
OR
visible crack line
```

Evaluation should determine whether this rule behaves as intended in cases containing structural-crack features.

Particular attention should be given to cases with:

* pain on biting;
* pain on release;
* poor pain localization;
* visible crack lines;
* concurrent periapical radiographic abnormalities.

---

# 16. Standard Endodontic-Origin Pathway

The third Stage 7.2 gate requires:

```text
cold response = FALSE
AND
EPT response = FALSE
```

and is considered only after the preceding fracture-related gates are negative.

Evaluation should therefore preserve the priority hierarchy rather than evaluating this condition in isolation.

---

# 17. Non-Endodontic Red-Flag Pathway

The fourth gate is:

```text
(
    cold response = TRUE
    OR
    EPT response = TRUE
)
AND
periapical lesion present
```

The system labels this:

```text
NON_ENDODONTIC_RED_FLAG
```

Evaluation should assess:

* sensitivity for cases requiring non-endodontic consideration;
* false-positive red flags;
* false-negative red flags;
* interaction with fracture-related cases.

Importantly, the v65 system does not automatically classify every radiographic/clinical discordance as a non-endodontic diagnosis.

---

# 18. Indeterminate Cases

Cases that do not satisfy a higher-priority gate remain:

```text
INDETERMINATE
```

This category should be retained during evaluation.

An indeterminate result should not automatically be treated as a diagnostic error.

Instead, evaluation should distinguish:

```text
Appropriate uncertainty
vs.
Incorrect classification
```

when an appropriate reference standard is available.

---

# 19. RAG Evaluation

ENDOLLM uses:

```text
BAAI/bge-small-en-v1.5
```

for retrieval.

The default retrieval depth is:

```text
TOP_K = 5
```

The retrieval query incorporates target-specific clinical and radiographic information.

Evaluation of retrieval should assess whether the retrieved evidence is:

* relevant;
* clinically appropriate;
* target-specific;
* supportive of the diagnostic context;
* free from obvious irrelevant material.

If formal retrieval metrics are reported, the repository should document:

```text
Query set
Relevant-document definition
Ground-truth evidence
Recall@K
Precision@K
MRR / other metric, if used
```

These metrics should only be reported if actually calculated.

---

# 20. LLM Reasoning Evaluation

The Qwen2.5-VL model is used for evidence-grounded reasoning.

It is not intended to replace the deterministic diagnostic framework.

Therefore, LLM evaluation should focus on:

* factual consistency with structured clinical findings;
* consistency with radiographic observations;
* adherence to target-tooth grounding;
* adherence to the Stage 7.2 hierarchy;
* correct use of retrieved evidence;
* avoidance of unsupported diagnoses;
* appropriate expression of uncertainty;
* appropriate escalation.

The model should not receive credit merely for producing a plausible-sounding narrative.

---

# 21. Programmatic-vs-LLM Agreement

An important evaluation dimension is whether the generated narrative remains consistent with the authoritative programmatic state.

The v65 architecture establishes:

```text
Programmatic diagnostic labels
        +
Programmatic triage
        +
Programmatic red flags
        +
Programmatic escalation
        ↓
Authoritative clinical framework
        ↓
LLM explanation
```

Evaluation should therefore assess whether the LLM:

1. preserves the programmatic diagnosis;
2. preserves the lesion-origin classification;
3. preserves escalation requirements;
4. avoids contradicting deterministic safety rules;
5. accurately explains the evidence.

---

# 22. Safety Evaluation

The v65 implementation contains explicit response validation.

The validator checks for problematic statements such as inappropriate claims that a positive cold response:

```text
confirms pulp vitality
proves pulp vitality
indicates a vital pulp
```

The safety layer also checks for inappropriate interpretation of PAI as a direct clinical diagnosis.

Safety evaluation should therefore record:

```text
Number of unsafe outputs
Number of corrected outputs
Number of missed unsafe outputs
Number of false-positive safety flags
```

if these measures are formally evaluated.

---

# 23. Diagnostic Label Evaluation

The v65 system produces probable pulpal/apical diagnostic labels based on diagnostic consistency.

Evaluation should compare the system's final structured labels with an appropriately defined clinician/reference standard.

The evaluation should distinguish:

```text
Pulpal diagnosis
Apical diagnosis
Lesion-origin classification
Triage classification
Escalation requirement
```

These are related but distinct outputs and should not be collapsed into a single score without justification.

---

# 24. Structured Output Evaluation

Each case generates a structured JSON audit record.

The output includes fields corresponding to:

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

Evaluation should verify that:

* required fields are present;
* data types are valid;
* target-tooth information is preserved;
* programmatic decisions are preserved;
* evidence references are retained;
* safety validation is recorded.

---

# 25. Auditability Evaluation

One of the objectives of the structured architecture is to make the diagnostic process inspectable.

For each case, an evaluator should be able to trace:

```text
Input image
    ↓
Visual findings
    ↓
Target tooth
    ↓
Clinical history
    ↓
Structured clinical findings
    ↓
Diagnostic consistency
    ↓
Lesion-origin classification
    ↓
Triage
    ↓
Retrieved evidence
    ↓
LLM reasoning
    ↓
Safety validation
    ↓
Final structured output
```

An error-analysis study can therefore identify whether an incorrect result originated from:

* visual detection;
* tooth association;
* clinical input;
* structured extraction;
* deterministic rule;
* retrieval;
* LLM reasoning;
* safety validation.

---

# 26. Error Analysis

Performance evaluation should not rely exclusively on aggregate accuracy.

Important error categories include:

### Visual errors

* missed lesion;
* false-positive lesion;
* incorrect localization;
* incorrect tooth association.

### Clinical extraction errors

* missed positive finding;
* incorrect negative finding;
* unknown incorrectly converted to negative.

### Diagnostic-rule errors

* inappropriate rule activation;
* incorrect precedence;
* inappropriate indeterminate classification.

### Retrieval errors

* irrelevant evidence;
* insufficient evidence;
* incorrect target context.

### LLM errors

* unsupported diagnosis;
* contradiction of programmatic classification;
* target-tooth drift;
* incorrect interpretation of evidence;
* excessive certainty.

### Safety errors

* inappropriate vitality claim;
* inappropriate PAI interpretation;
* failure to preserve escalation;
* unsupported rare diagnosis.

---

# 27. Evaluation of Target Locking

A specific evaluation should assess whether the system maintains the clinician-selected target tooth throughout the complete pipeline.

A target-locking failure occurs if:

```text
Target tooth = A

but final reasoning uses a finding belonging to

Tooth B
```

Evaluation should therefore compare:

```text
Clinician-confirmed target
        vs.
Target-associated evidence
        vs.
Final diagnostic reasoning
```

This is particularly important in radiographs containing multiple abnormalities.

---

# 28. Evaluation of Uncertainty

ENDOLLM explicitly preserves uncertainty.

Examples include:

```text
UNKNOWN / NONE
INDETERMINATE
Confidence: indeterminate
Specialist verification required
```

Evaluation should therefore assess whether the system appropriately identifies insufficient information rather than forcing a definitive diagnosis.

This is an important component of clinical safety evaluation.

---

# 29. Specialist Verification

The v65 output explicitly retains a specialist-verification requirement.

The system should therefore be evaluated not only for diagnostic agreement but also for whether it appropriately identifies cases requiring additional clinical assessment.

Examples may include cases involving:

* suspected vertical root fracture;
* suspected cracked tooth;
* non-endodontic red flags;
* insufficient clinical information;
* advanced-imaging escalation;
* diagnostic discordance.

The exact reference standard for escalation should be defined in the corresponding clinical evaluation protocol.

---

# 30. Statistical Analysis

The appropriate statistical analysis depends on the evaluation endpoint.

For categorical diagnostic outputs, the evaluation may include measures such as:

```text
Sensitivity
Specificity
Positive predictive value
Negative predictive value
Accuracy
F1 score
Cohen's kappa
Confusion matrices
```

For continuous or ordinal outcomes, appropriate descriptive and inferential statistics should be selected according to the study design and distribution.

For multiple-model comparisons, the statistical procedure should account for:

* paired observations where applicable;
* repeated evaluation of the same radiographs;
* non-normal distributions;
* multiple comparisons.

The exact statistical methods used in the manuscript should be documented in the manuscript-associated analysis files rather than inferred from the v65 inference code.

---

# 31. Reproducibility Requirements

A reproducible evaluation should specify:

```text
Dataset version
Train/validation/test split
Model versions
Checkpoint identifiers
Inference parameters
Retrieval index version
Knowledge-base version
Prompt/version
Diagnostic-rule version
Software environment
Evaluation script
Statistical analysis script
Random seeds where applicable
```

The public repository should maintain the manuscript-associated versions of these components wherever redistribution is permitted.

---

# 32. Version Control

Evaluation results should always be linked to the corresponding ENDOLLM version.

For example:

```text
ENDOLLM v65
```

should not be mixed with results generated from a later development version unless the difference is explicitly documented.

Changes to:

* diagnostic rules;
* prompts;
* retrieval index;
* model checkpoints;
* visual models;
* preprocessing;
* target-tooth association;
* safety validation;

may alter system behavior and should therefore result in a new version or clearly documented revision.

---

# 33. Current v65 Evaluation Scope

The current v65 implementation supports evaluation of:

```text
Radiographic perception
Target-tooth grounding
Clinical information extraction
Diagnostic consistency
Lesion-origin classification
Triage
Retrieval
LLM reasoning
Safety validation
Structured output
```

The v65 source itself does not constitute a complete statistical evaluation script.

Therefore, the final repository should add dedicated evaluation scripts/results when those analyses are available.

---

# 34. Evaluation Reporting Principles

The ENDOLLM evaluation should distinguish between:

### Model performance

Performance of an individual visual or computational model.

### Rule performance

Performance of deterministic clinical rules.

### Retrieval performance

Quality of evidence retrieval.

### LLM performance

Quality and safety of generated reasoning.

### System performance

Performance of the integrated end-to-end pipeline.

These should not be conflated into a single measure without explicitly defining the methodology.

---

# 35. Recommended Evaluation Structure

The repository may eventually contain:

```text
evaluation/
│
├── scripts/
│   ├── evaluate_vision.py
│   ├── evaluate_target_grounding.py
│   ├── evaluate_diagnostic_rules.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_llm_outputs.py
│   └── evaluate_safety.py
│
├── results/
│
├── figures/
│
└── tables/
```

This is a recommended organizational structure.

It does not imply that these scripts currently exist in the v65 repository.

---

# 36. Relationship to Manuscript Results

The manuscript should report results generated from a clearly identified version of the ENDOLLM system.

The repository should make it possible to establish:

```text
Manuscript result
        ↓
Evaluation dataset
        ↓
Model/checkpoint
        ↓
ENDOLLM version
        ↓
Evaluation procedure
        ↓
Statistical analysis
```

This linkage is particularly important for reviewer reproducibility.

---

# 37. Limitations of Evaluation

Several limitations should be considered.

### Upstream model dependence

End-to-end performance depends on the accuracy of the underlying frozen visual models.

### Clinical-input dependence

The system receives clinician-provided clinical information. Errors or omissions in the clinical
