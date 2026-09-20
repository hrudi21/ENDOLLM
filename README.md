# ENDOLLM
ENDOLLM: An interactive multimodal large language model framework for endodontic diagnostic decision support.
# ENDOLLM

## An Interactive Multimodal Large Language Model Framework for Endodontic Diagnostic Reasoning

**Current version:** Phase 4.5 Interactive Endodontic MLLM (v65) — Stage 7.2
**Project stage:** 7/9
**Status:** Research prototype / clinician-supervised decision-support system

---

## Overview

**ENDOLLM (Endodontic Large Language Model)** is a multimodal, interactive artificial intelligence framework developed for **endodontic diagnostic decision support**.

The current implementation, **Phase 4.5 Interactive Endodontic MLLM (v65)**, integrates:

* frozen radiographic image-analysis models;
* tooth-level and panoramic visual analysis through OPGAgent;
* interactive clinician identification of the symptomatic target tooth;
* structured clinical history elicitation;
* deterministic clinical diagnostic rules;
* fracture and lesion-origin assessment;
* retrieval-augmented generation (RAG);
* a multimodal large language model for evidence-grounded reasoning;
* structured diagnostic outputs;
* safety validation and escalation logic;
* an auditable JSON representation of the complete diagnostic process.

A central design principle of ENDOLLM is that the language model is **not the primary diagnostic decision maker**.

Instead, the system combines programmatic clinical rules, frozen visual models, structured clinical information, retrieved evidence, and clinician verification. The LLM is used primarily for **interactive history taking, evidence synthesis, and explanation**, while predefined diagnostic and safety rules remain authoritative.

---

# 1. System Design Philosophy

ENDOLLM follows a **hybrid AI architecture** rather than relying on unconstrained LLM diagnosis.

The current v65 pipeline separates:

1. **Radiographic perception**
2. **Tooth identification and visual findings**
3. **Clinician target-tooth selection**
4. **Clinical history acquisition**
5. **Clinical information normalization**
6. **Deterministic diagnostic consistency assessment**
7. **Lesion-origin and fracture assessment**
8. **Safety and red-flag triage**
9. **Knowledge retrieval**
10. **LLM-based evidence synthesis**
11. **Structured output generation**
12. **Safety validation and audit logging**

This separation is intentional. The programmatic diagnostic labels, triage decisions, lesion-origin classification, and escalation rules are treated as authoritative within the system, while the LLM is constrained to operate around those decisions.

---

# 2. Current v65 Architecture

The current implementation is designated:

> **Phase 4.5 Interactive Endodontic MLLM (v65) — Stage 7.2**

The principal components are:

```text
                         INPUT RADIOGRAPH
                                │
                                ▼
                  ┌──────────────────────────┐
                  │ Frozen Vision Pipeline   │
                  │                          │
                  │ Faster R-CNN            │
                  │ ResNet-50               │
                  └────────────┬─────────────┘
                               │
                               ▼
                    Radiographic Findings
                               │
                               │
                 ┌─────────────▼─────────────┐
                 │       OPGAgent            │
                 │                           │
                 │ Tooth enumeration         │
                 │ Quadrants                 │
                 │ Disease/finding detection │
                 │ Bone-loss assessment      │
                 └─────────────┬─────────────┘
                               │
                               ▼
                     Tooth-level Findings
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Interactive Target-Tooth   │
                 │ Gate                       │
                 │                            │
                 │ Clinician identifies       │
                 │ symptomatic tooth/teeth    │
                 └─────────────┬──────────────┘
                               │
                               ▼
                    Target-specific evidence
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Clinical History            │
                 │ Elicitation                 │
                 │                            │
                 │ Qwen2.5-VL                 │
                 │ ≤5 focused questions       │
                 └─────────────┬──────────────┘
                               │
                               ▼
                    Structured Clinical Data
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        Diagnostic Consistency       Stage 7.2
              Analysis            Lesion-Origin Rules
                  │                         │
                  └────────────┬────────────┘
                               ▼
                    Authoritative Triage
                               │
                               ▼
                  Dynamic Knowledge Retrieval
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Qwen2.5-VL-7B-Instruct  │
                 │                          │
                 │ Evidence-grounded       │
                 │ diagnostic reasoning    │
                 └────────────┬─────────────┘
                              │
                              ▼
                   Safety Validation
                              │
                              ▼
                    Structured JSON Output
```

---

# 3. Frozen Radiographic Vision Pipeline

The v65 implementation integrates a previously established/frozen radiographic vision pipeline.

The code identifies the following components:

* **FasterRCNN_epoch14**
* **ResNet50_epoch12**
* **2025 AAE/ESE periapical terminology framework**

The frozen vision pipeline provides radiographic observations that are passed into the interactive diagnostic workflow.

The v65 code explicitly treats these findings as **observations** rather than allowing the language model to freely reinterpret the underlying image without constraint.

The frozen pipeline is located externally to the main v65 script and is imported from the ENDOLLM project root.

---

# 4. OPGAgent Integration

ENDOLLM v65 also integrates **OPGAgent** for additional visual and tooth-level analysis.

The current code references:

```text
OPGAgent/
```

and uses:

```text
api_service/yolo_enumeration/model/best.pt
```

for tooth enumeration.

The OPGAgent component includes visual analysis for:

* tooth enumeration;
* quadrant identification;
* disease/finding detection;
* bone-loss assessment;
* tooth-level association of detected findings.

The current implementation references TVEM model configurations/checkpoints for:

```text
11diseases
4quadrants
mandibular_maxillary
bone_loss
```

The configured confidence thresholds in v65 are:

```text
Tooth enumeration: 0.25
TVEM findings:      0.30
```

These values are implementation parameters in the current code.

---

# 5. Target-Tooth Grounding

A major feature of v65 is the **interactive target-tooth gate**.

Rather than allowing every radiographic finding in an image to influence the final diagnostic reasoning, the system requires the clinician to identify the **symptomatic tooth or teeth**.

The workflow is:

```text
Radiographic observations
        ↓
OPGAgent tooth-level observations
        ↓
Clinician identifies symptomatic tooth/teeth
        ↓
System validates selected tooth
        ↓
Target-specific findings are assembled
        ↓
Only target-associated findings proceed
        ↓
Diagnostic reasoning
```

This mechanism is intended to reduce diagnostic contamination from unrelated findings elsewhere in the radiograph.

The clinician is shown visual findings as observations and is asked to identify the symptomatic tooth/teeth using human-readable tooth names.

The system then requests explicit confirmation.

---

# 6. Target-Specific Evidence Association

The v65 implementation performs spatial association between detected lesions/findings and teeth.

The code uses:

* bounding-box intersection-over-union (IoU);
* lesion containment;
* tooth-level association.

The configured thresholds include:

```text
Minimum IoU:                0.05
Minimum lesion containment: 0.50
```

Only findings associated with the clinician-selected target tooth/teeth are treated as target-specific diagnostic evidence.

Non-tooth-level findings may remain available as contextual information but are not automatically treated as target-specific evidence.

This target-locking mechanism is an important component of the current architecture.

---

# 7. Interactive Clinical History

After target-tooth selection, Qwen2.5-VL is used to generate a limited number of focused clinical questions.

The v65 implementation allows a maximum of:

```text
5 focused questions
```

The questions address clinically relevant features including:

* chief complaint;
* symptom onset and duration;
* pain on biting;
* pain on release;
* difficulty localizing pain;
* cold response;
* cold lingering;
* electric pulp testing;
* percussion;
* palpation;
* spontaneous pain;
* swelling;
* sinus tract;
* periodontal probing;
* isolated deep/narrow pockets;
* mobility;
* furcation involvement;
* suspected or visible cracks;
* previous root canal treatment;
* trauma history;
* systemic infection signs.

The question-generation mechanism is designed to gather missing clinical information rather than directly produce a diagnosis.

---

# 8. Structured Clinical Representation

Free-text clinical information is normalized into a structured representation before diagnostic reasoning.

The system uses three-state logic:

```text
TRUE
FALSE
UNKNOWN / NONE
```

An unknown clinical finding is **not automatically treated as negative**.

This distinction is important because missing clinical information should not be converted into a false clinical finding.

The structured representation includes variables such as:

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

The implementation also normalizes common clinical terminology, synonyms, and common typographical variations.

---

# 9. Stage 7.2 Deterministic Lesion-Origin Classification

The current v65 implementation introduces a deterministic lesion-origin hierarchy designated **Stage 7.2**.

The hierarchy is explicitly ordered:

```text
Priority 1 — Vertical root fracture suspected
Priority 2 — Cracked tooth suspected
Priority 3 — Standard endodontic origin supported
Priority 4 — Non-endodontic red flag
Priority 5 — Indeterminate
```

The purpose of this hierarchy is to prevent the presence of a periapical radiographic lesion from automatically forcing the case into a conventional endodontic diagnosis.

---

## 9.1 Gate 1 — Vertical Root Fracture

A vertical root fracture pathway is triggered when:

```text
Previously treated
AND
(
    isolated narrow deep pocket
    OR coronal/lateral sinus tract
    OR J-shaped / halo lesion morphology
)
```

The resulting category is:

```text
VRF_SUSPECTED
```

The v65 implementation assigns:

```text
Priority: 1
Confidence: indeterminate
```

---

## 9.2 Gate 2 — Cracked Tooth

The cracked-tooth pathway is evaluated only when the VRF gate is negative.

The rule is:

```text
(
    pain on release
    AND
    difficulty localizing pain
)
OR
visible crack line
```

The resulting category is:

```text
CRACKED_TOOTH_SUSPECTED
```

with:

```text
Priority: 2
```

---

## 9.3 Gate 3 — Standard Endodontic Origin

The conventional endodontic-origin pathway is evaluated only after the fracture-related gates are negative.

The rule requires:

```text
cold response = FALSE
AND
EPT response = FALSE
```

The resulting category is:

```text
ENDODONTIC_ORIGIN_SUPPORTED
```

with:

```text
Priority: 3
Confidence: supported
```

---

## 9.4 Gate 4 — Non-Endodontic Red Flag

The non-endodontic red-flag pathway is evaluated only after the preceding gates are negative.

The implemented condition is:

```text
(
    cold response = TRUE
    OR
    EPT response = TRUE
)
AND
periapical lesion present
```

The resulting category is:

```text
NON_ENDODONTIC_RED_FLAG
```

---

## 9.5 Gate 5 — Indeterminate

If none of the preceding rules are satisfied, the case remains:

```text
INDETERMINATE
```

The implementation deliberately preserves uncertainty when required clinical information is unavailable.

---

# 10. Separation of Biological Discordance and Lesion Origin

An important design feature of v65 is that **biological/periapical discordance is not automatically equated with a non-endodontic diagnosis**.

For example, cases involving:

* suspected vertical root fracture;
* suspected cracked tooth;

retain their fracture-related classification even when a periapical lesion is present.

The code preserves these states separately for auditability.

This prevents the presence of a radiographic lesion from overriding the higher-priority fracture pathway.

---

# 11. Diagnostic Consistency and Triage

The system performs diagnostic consistency assessment before final LLM reasoning.

The v65 pipeline integrates:

```text
evaluate_diagnostic_consistency(...)
```

and:

```text
assess_non_endodontic_red_flags(...)
```

from the frozen project components.

The Stage 7.2 lesion-origin classifier is then used to establish the authoritative triage state.

The merged triage representation retains:

* category;
* priority;
* gate;
* confidence;
* basis;
* reason;
* previous category where relevant;
* biological/periapical discordance;
* non-endodontic red-flag status.

The Stage 7.2 classification is explicitly treated as authoritative over conflicting downstream language-model interpretations.

---

# 12. Dynamic Clarification and Safety Logic

The system can dynamically identify areas where additional clarification is needed.

Examples include:

### Positive cold response

A positive cold response is not automatically converted into a definitive diagnosis of pulp vitality.

### Structural or fracture concerns

Structural concerns preserve the Stage 7.2 hierarchy.

### Isolated deep periodontal pocket

The system can direct the reasoning toward a periodontal/fracture pathway.

### Biting or release pain

Pain on biting or release is not automatically treated as atypical. The v65 logic allows this symptom to be compatible with:

* periradicular inflammation;
* structural crack.

---

# 13. Retrieval-Augmented Generation

ENDOLLM v65 incorporates a retrieval-augmented generation pipeline.

The retrieval model is:

```text
BAAI/bge-small-en-v1.5
```

The implementation uses normalized embeddings and dot-product similarity to retrieve relevant knowledge chunks.

The default retrieval depth is:

```text
TOP_K = 5
```

The current code references the following knowledge-base files:

```text
bge_small_embeddings.npy
bge_small_metadata.jsonl
chunks.jsonl
```

and merged versions:

```text
bge_small_embeddings_merged_244.npy
bge_small_metadata_merged_244.jsonl
chunks_merged_244.jsonl
```

The retrieval system stores information including:

* retrieval rank;
* similarity score;
* source file;
* page information;
* chunk ID;
* retrieved text.

---

# 14. Dynamic Retrieval Query Construction

Retrieval is not based solely on the radiographic image.

The v65 retrieval query can incorporate:

* target tooth;
* target-associated frozen radiographic findings;
* target-associated OPGAgent findings;
* verified structured clinical findings;
* relevant diagnostic themes.

The retrieval logic includes targeted pathways for:

* previous root canal treatment;
* cracked tooth/release pain;
* vertical root fracture;
* symptomatic apical periodontitis;
* other clinically relevant endodontic concepts.

The retrieval logic also contains safeguards so that fracture-related priority is not inadvertently replaced by a generic pain-on-biting retrieval pathway when the VRF pathway has already been activated.

---

# 15. Multimodal Language Model

The current v65 implementation uses:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

The model is loaded using 4-bit quantization.

The configured quantization approach includes:

```text
NF4
Double quantization
Float16 computation
Automatic device mapping
```

Generation is configured for deterministic-style output using:

```text
do_sample = False
```

with a repetition penalty of:

```text
1.05
```

The maximum generated token count is configurable through the implementation.

---

# 16. Role of the LLM

The LLM has a deliberately constrained role.

It is used for:

* interactive clinical history elicitation;
* interpretation of structured clinical information;
* synthesis of radiographic observations;
* synthesis of OPGAgent findings;
* evidence retrieval integration;
* diagnostic reasoning narrative;
* generation of clinician-readable explanations.

The LLM is **not intended to independently override programmatic diagnostic rules**.

The final prompt explicitly informs the model that:

* programmatic diagnostic labels are authoritative;
* triage and red-flag classifications are authoritative;
* escalation decisions are authoritative;
* the target tooth is locked by clinician confirmation;
* unsupported diagnoses should not be invented;
* rare non-endodontic neoplasm names should not be generated without appropriate evidence;
* PAI/radiographic indices should not be misrepresented as clinical diagnoses.

---

# 17. Diagnostic Label Selection

The v65 implementation selects probable pulpal and apical diagnostic labels using the diagnostic consistency results.

The system also considers the authoritative lesion-origin classification.

When the authoritative origin is:

```text
VRF_SUSPECTED
```

the apical diagnosis is set to:

```text
indeterminate
```

rather than forcing a conventional apical diagnosis.

This preserves the distinction between radiographic observations and the underlying clinical diagnostic process.

---

# 18. Evidence-Grounded Diagnostic Reasoning

The final reasoning stage receives a structured, target-locked clinical and radiographic context.

The reasoning framework emphasizes:

1. target-tooth identity;
2. frozen radiographic observations;
3. target-associated OPGAgent findings;
4. structured clinical findings;
5. diagnostic consistency;
6. authoritative lesion-origin classification;
7. triage/red-flag status;
8. retrieved evidence;
9. uncertainties;
10. escalation requirements.

The final response is structured under the following headings:

```text
1. Triage conclusion
2. Radiographic findings
3. Diagnostic reasoning
4. Evidence supporting the interpretation
5. Missing or uncertain information
```

The system then appends evidence-source references and a specialist-verification disclaimer.

---

# 19. Safety Validation

Before the final result is returned, the generated response passes through a validation layer.

The validator checks for inappropriate claims, including examples such as:

```text
"confirms pulp vitality"
"proves pulp vitality"
```

and other potentially misleading diagnostic statements.

The validation framework also protects against inappropriate use of radiographic indices, including misuse of the PAI as a direct clinical diagnosis.

---

# 20. Structured Audit Output

The system produces a structured JSON record for each processed case.

The output contains information such as:

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

This structured representation is intended to support:

* reproducibility;
* auditing;
* error analysis;
* research evaluation;
* manuscript reporting;
* inspection of intermediate decisions.

---

# 21. Output Directory

The v65 implementation writes structured diagnostic outputs to:

```text
phase2_knowledge/evaluation/phase4_outputs/
```

The output filename follows the pattern:

```text
{image_stem}_phase4_structured_diagnostic.json
```

For example, an input image with a filename corresponding to a case can generate a structured JSON record whose filename preserves the image stem.

---

# 22. End-to-End Workflow

The current v65 execution sequence is approximately:

```text
1. Load input radiograph
        ↓
2. Run frozen radiographic vision
        ↓
3. Run OPGAgent visual analysis
        ↓
4. Display/assemble radiographic observations
        ↓
5. Ask clinician to identify target tooth/teeth
        ↓
6. Validate target tooth
        ↓
7. Associate target-specific visual findings
        ↓
8. Load Qwen2.5-VL
        ↓
9. Generate focused clinical questions
        ↓
10. Clinician provides clinical history
        ↓
11. Normalize clinical terminology
        ↓
12. Extract structured clinical findings
        ↓
13. Evaluate diagnostic consistency
        ↓
14. Assess non-endodontic red flags
        ↓
15. Apply Stage 7.2 lesion-origin hierarchy
        ↓
16. Merge authoritative triage
        ↓
17. Select diagnostic labels
        ↓
18. Align escalation
        ↓
19. Construct dynamic retrieval query
        ↓
20. Retrieve evidence from knowledge base
        ↓
21. Generate evidence-grounded reasoning
        ↓
22. Validate LLM response
        ↓
23. Build structured JSON audit output
        ↓
24. Save output
```

---

# 23. Command-Line Input

The current v65 script accepts a radiographic image path as a positional command-line argument.

Conceptually:

```bash
python phase4_interactive_endodontic_mllm_v65.py <image_path>
```

The retrieval depth can also be controlled using:

```text
--top-k
```

with a default value of:

```text
5
```

Example:

```bash
python phase4_interactive_endodontic_mllm_v65.py /path/to/radiograph.jpg --top-k 5
```

**Note:** The exact production installation command and environment configuration should be documented separately once the repository structure is finalized.

---

# 24. Repository Dependencies and External Components

The v65 script references several external components.

These include:

### ENDOLLM project root

The code imports the frozen diagnostic and vision components from the project root.

Referenced components include:

```text
evaluate_diagnostic_consistency
assess_non_endodontic_red_flags
analyze_radiograph
```

### OPGAgent

The code references:

```text
OPGAgent/
```

including the tooth enumeration model and TVEM components.

### Knowledge base

The RAG system expects the knowledge-base directory:

```text
EndoMLLM/phase2_knowledge/
```

### Model

The current language model is:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

### Retrieval encoder

The current retrieval encoder is:

```text
BAAI/bge-small-en-v1.5
```

Exact installation instructions, model download procedures, licenses, and redistribution permissions should be documented in the repository after verification.

---

# 25. Reproducibility Considerations

The current v65 code was developed as part of an evolving research system.

The script currently references machine-relative locations using:

```python
Path.home()
```

for example:

```text
~/EndoMLLM/
~/OPGAgent/
~/opgagent_env/
```

Therefore, a clean public repository release should convert these assumptions into configurable paths or environment variables.

Before public release, the following should be checked:

* removal of machine-specific assumptions;
* reproducible environment specification;
* exact Python version;
* dependency versions;
* model availability;
* model licensing;
* dataset licensing;
* external repository references;
* retrieval-index generation procedure;
* checkpoint provenance;
* output directory creation;
* `.env` handling if credentials are required.

The v65 script itself does not establish the final licensing status of all external components. Those licenses should therefore be verified before redistribution.

---

# 26. Data and Dataset Policy

The repository should distinguish between:

### Code developed for ENDOLLM

This should be version-controlled where appropriate.

### Public datasets

The repository should provide:

* dataset name;
* original source;
* publication/reference;
* official URL;
* license;
* preprocessing procedure;
* subset used;
* train/validation/test split where applicable.

The dataset files themselves should only be redistributed when their respective licenses permit redistribution.

### Third-party model weights

Model weights should not automatically be committed to the repository.

Where redistribution is restricted or unnecessary, the repository should provide:

* official model name;
* official source;
* version/checkpoint;
* download instructions;
* license.

### Derived embeddings

The repository should document whether the RAG embeddings and merged indexes can legally and practically be redistributed.

These decisions will be finalized in the repository's dataset and reproducibility documentation.

---

# 27. Clinical Safety and Intended Use

ENDOLLM is a **research prototype for clinician-supervised endodontic decision support**.

It is not intended to replace:

* clinical examination;
* pulp sensibility testing;
* pulp testing;
* periodontal probing;
* percussion/palpation;
* crack assessment;
* radiographic interpretation by a clinician;
* CBCT or other advanced imaging when clinically indicated;
* specialist consultation;
* definitive clinical diagnosis.

The system deliberately requires clinician interaction and verification.

The current architecture explicitly includes escalation and specialist-verification requirements where the available information is insufficient or when higher-risk diagnostic pathways are encountered.

---

# 28. Important Limitations

The current v65 implementation has several limitations that should be considered when interpreting research results.

### 28.1 Research prototype

The system is a research-stage implementation and should not be interpreted as a validated autonomous clinical diagnostic system.

### 28.2 Dependence on upstream models

The final system is dependent on the performance and limitations of the frozen radiographic vision pipeline and OPGAgent.

### 28.3 Target-tooth identification

Target-specific reasoning depends on accurate clinician identification and confirmation of the symptomatic tooth/teeth.

### 28.4 Clinical input quality

Structured reasoning depends on the accuracy and completeness of clinician-provided history and examination findings.

### 28.5 Rule-based boundaries

The Stage 7.2 hierarchy represents explicitly programmed diagnostic logic and should be evaluated against appropriately designed clinical reference standards.

### 28.6 Retrieval dependency

Evidence-grounded reasoning depends on the quality, completeness, and relevance of the underlying knowledge base.

### 28.7 LLM limitations

Qwen2.5-VL may still generate incorrect or incomplete explanations. Programmatic safeguards reduce but do not eliminate this risk.

### 28.8 External dependencies

The current implementation depends on external models, repositories, checkpoints, and knowledge-base files.

### 28.9 Generalizability

Performance should not be assumed to generalize beyond the populations, imaging conditions, datasets, clinical scenarios, and terminology represented in the evaluation data.

---

# 29. Transparency and Auditability

A major design goal of ENDOLLM is to make the diagnostic process inspectable.

Rather than storing only a final textual diagnosis, the system records intermediate information such as:

```text
Input image
        ↓
Radiographic findings
        ↓
Tooth-level findings
        ↓
Target tooth selected
        ↓
Clinical history
        ↓
Structured clinical profile
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

This allows investigators to examine where an incorrect or uncertain result originated.

---

# 30. Research Reproducibility

For manuscript-associated reproducibility, the repository is intended to document:

* the exact v65 source code;
* architecture;
* model versions;
* diagnostic rules;
* retrieval system;
* datasets and their sources;
* preprocessing;
* evaluation procedures;
* structured output format;
* known limitations.

The public repository should contain enough information for an independent researcher to understand the system architecture and reproduce the computational workflow to the extent permitted by third-party dataset and model licenses.

---

# 31. Version Information

Current implementation:

```text
ENDOLLM
Phase 4.5 Interactive Endodontic MLLM
Version: v65
Stage: 7.2
Project workflow stage: 7/9
```

The v65 implementation should be treated as a specific research snapshot.

Future modifications should be versioned rather than silently replacing the manuscript-associated implementation.

---

# 32. Recommended Repository Structure

The public repository is intended to evolve toward a structure similar to:

```text
ENDOLLM/
│
├── README.md
├── DATASET.md
├── TRAINING.md
├── EVALUATION.md
├── ARCHITECTURE.md
├── MANUSCRIPT_REPRODUCIBILITY.md
├── MODEL_CARD.md
├── CITATION.cff
├── LICENSE
├── .gitignore
├── .env.example
│
├── src/
│   ├── phase4_interactive_endodontic_mllm_v65.py
│   ├── diagnostic_rules.py
│   ├── non_endodontic_red_flags.py
│   └── vision_pipeline.py
│
├── phase2_knowledge/
│   └── evaluation/
│       └── phase4_outputs/
│
├── configs/
│
├── scripts/
│
└── results/
```

This is a **recommended repository organization**, not a claim that the current v65 file already has this exact directory structure.

---

# 33. What This Repository Will Provide

The repository is intended to provide researchers and reviewers with:

### Architecture transparency

A clear description of how image analysis, clinical information, deterministic rules, retrieval, and LLM reasoning interact.

### Code transparency

The manuscript-associated implementation can be versioned and inspected.

### Diagnostic-rule transparency

The Stage 7.2 hierarchy can be independently examined.

### Evidence transparency

Retrieved evidence and evidence references can be represented in structured outputs.

### Auditability

Intermediate decisions can be inspected rather than only the final generated response.

### Reproducibility

Researchers can reconstruct the computational workflow using the documented models, datasets, configurations, and procedures, subject to third-party licensing restrictions.

---

# 34. Citation

A formal citation record will be added to the repository once the manuscript and repository bibliographic information are finalized.

A `CITATION.cff` file will accompany the repository.

---

# 35. License

The final repository license is **to be determined** after verification of:

* authorship;
* institutional requirements;
* third-party model licenses;
* dataset licenses;
* external repository licenses;
* redistribution permissions.

No license should be inferred solely from the v65 source code.

---

# 36. Disclaimer

ENDOLLM is a research prototype developed for investigation of multimodal artificial intelligence in endodontic diagnostic decision support.

The system does not replace professional clinical judgment.

All generated interpretations must be reviewed and verified by an appropriately qualified dental professional before any clinical decision is made.

---

# 37. Current Development Stage

**ENDOLLM project workflow: Stage 7/9**

The current repository preparation corresponds to the **research reproducibility and public-code preparation stage**.

The v65 implementation specifically represents:

> **Phase 4.5 Interactive Endodontic MLLM (v65) — Stage 7.2**

The distinction is intentional:

* **7/9** = overall ENDOLLM project workflow stage.
* **Stage 7.2** = diagnostic lesion-origin rule stage implemented within v65.

---

## Acknowledgement of Current v65 Implementation

This README is based on the architecture and implementation contained in:

`phase4_interactive_endodontic_mllm_v65.py`

The repository documentation should be updated whenever the manuscript-associated implementation changes materially.

**Version-controlled research code should remain synchronized with the version described in the associated manuscript.**
