# ENDOLLM Dataset Documentation

## Dataset Overview

The ENDOLLM project uses a structured radiographic dataset developed for multimodal endodontic AI research.

The dataset contains:

* **3,924 radiographs**
* **11,772 structured JSONL examples**
* Three task types for each image:

  * classification
  * lesion counting
  * lesion localization

The localization annotations contain **periapical-lesion bounding boxes** together with **Periapical Index (PAI) classes 3–5**.

Bounding-box coordinates are represented in the **original-image pixel coordinate system**.

---

## 1. Dataset Composition

The complete dataset contains:

| Split      | Radiographs | JSONL examples |
| ---------- | ----------: | -------------: |
| Training   |       2,747 |          8,241 |
| Validation |         588 |          1,764 |
| Test       |         589 |          1,767 |
| **Total**  |   **3,924** |     **11,772** |

Each radiograph is represented by three task-specific examples:

1. **Classification**
2. **Counting**
3. **Localization**

Therefore:

```text
3,924 images × 3 task examples/image
= 11,772 JSONL examples
```

---

## 2. Dataset Splits

### Training Set

The training set contains:

```text
2,747 radiographs
8,241 JSONL records
```

The training records provide the image-associated examples used for model development.

---

### Validation Set

The validation set contains:

```text
588 radiographs
1,764 JSONL records
```

The validation set corresponds to the same three-task structure:

```text
classification
counting
localization
```

---

### Test Set

The test set contains:

```text
589 radiographs
1,767 JSONL records
```

The test set is maintained as a separate evaluation partition.

The verified dataset organization contains no missing image references between the structured records and their corresponding radiographic data.

---

# 3. Task Structure

Each image is represented through three complementary learning/evaluation tasks.

## 3.1 Classification

The classification task represents the presence/absence or diagnostic classification information associated with the radiograph according to the dataset annotation structure.

The exact class definitions should be interpreted from the corresponding annotation metadata and task files.

---

## 3.2 Counting

The counting task represents the number of annotated periapical lesions associated with the radiograph.

This provides a separate task from simple lesion presence/absence classification.

---

## 3.3 Localization

The localization task provides spatial information for annotated periapical lesions.

Localization annotations include:

* lesion bounding boxes;
* PAI class information;
* image-associated target information.

Bounding boxes are stored using coordinates corresponding to the **original radiographic image dimensions**.

---

# 4. Periapical Lesion Annotation

The dataset contains radiographs with annotated periapical lesions.

The localization annotations include periapical-lesion bounding boxes and PAI categories:

```text
PAI 3
PAI 4
PAI 5
```

The annotations are associated with the corresponding image metadata.

The inspected dataset structure confirms that localization targets correspond to the annotation metadata and that the target coordinates are represented in original-image pixel coordinates.

---

# 5. Dataset Files

The verified local ENDOLLM dataset organization contains the following principal files:

```text
dataset/
│
├── train_mllm.jsonl
├── validation_mllm.jsonl
├── test_mllm.jsonl
├── annotations_with_final_split.csv
└── image_level_metadata.csv
```

### `train_mllm.jsonl`

Contains the training examples.

Expected structure:

```text
classification
count
localization
```

for the training images.

---

### `validation_mllm.jsonl`

Contains the validation examples.

---

### `test_mllm.jsonl`

Contains the test examples.

The test partition should remain separate from model-development procedures when reproducing the reported evaluation.

---

### `annotations_with_final_split.csv`

Contains the annotation-level information together with the final dataset split assignment.

This file provides the linkage between the underlying annotations and the train/validation/test partitions.

---

### `image_level_metadata.csv`

Contains image-level metadata used to associate radiographs with their corresponding dataset information.

---

# 6. Data Integrity

The verified dataset contains:

```text
3,924 radiographs
11,772 JSONL examples
0 missing image references
```

The expected relationship is:

```text
1 radiograph
      │
      ├── classification example
      ├── counting example
      └── localization example
```

Therefore:

```text
3 examples/image × 3,924 images
= 11,772 examples
```

This correspondence was verified during dataset inspection.

---

# 7. Relationship to ENDOLLM

The dataset forms part of the radiographic AI development and evaluation framework underlying ENDOLLM.

The broader ENDOLLM system subsequently integrates radiographic visual analysis with:

* tooth-level visual information;
* clinician-selected target-tooth information;
* structured clinical history;
* deterministic diagnostic rules;
* lesion-origin classification;
* retrieval-augmented generation;
* multimodal language-model reasoning.

The dataset described here should therefore be distinguished from the **knowledge base used for retrieval-augmented generation**.

The RAG knowledge base is a separate project component and is not assumed to be identical to the radiographic dataset.

---

# 8. Relationship to the v65 System

The current public implementation is:

```text
Phase 4.5 Interactive Endodontic MLLM
v65
Stage 7.2
```

The v65 system references a separate knowledge-base directory:

```text
phase2_knowledge/
```

and uses retrieval files including:

```text
bge_small_embeddings.npy
bge_small_metadata.jsonl
chunks.jsonl
```

and merged versions of these files.

These retrieval resources should therefore not be confused with the radiographic dataset documented in this file.

---

# 9. External Components

ENDOLLM also incorporates external visual-analysis components, including the OPGAgent pipeline.

The v65 implementation references OPGAgent components for:

* tooth enumeration;
* quadrant identification;
* visual finding detection;
* bone-loss assessment.

These components are computational dependencies of the ENDOLLM system and are distinct from the radiographic dataset described in Sections 1–7.

The precise provenance, version, and licensing terms of each external component should be documented separately before redistribution.

---

# 10. Dataset Redistribution

The repository should distinguish between:

### Dataset documentation

This file documents the dataset structure and its role in ENDOLLM.

### Dataset images

The original radiographic image files should only be uploaded to this repository if their redistribution rights permit public distribution.

### Annotation files

Annotation files should only be redistributed after confirming that their underlying data and annotations may legally be shared.

### Derived files

Derived training files, embeddings, indexes, or transformed datasets should also be checked for redistribution permissions before being committed to the public repository.

For this reason, the initial public GitHub release should contain the **documentation and code first**, while the actual image dataset should not be uploaded until its redistribution status has been verified.

---

# 11. Dataset Provenance

The currently verified project records establish the dataset composition, annotation structure, and train/validation/test partitions described above.

The following provenance information should be added after verification:

```text
Original dataset name:
Original publication:
Dataset authors:
Source repository / website:
DOI:
License:
Date accessed:
Permission for redistribution:
```

These fields are intentionally left unresolved rather than populated with unverified information.

---

# 12. Reproducibility

To reproduce the dataset preparation, the repository should ultimately document:

1. Original dataset source.
2. Image inclusion criteria.
3. Annotation criteria.
4. PAI classification procedure.
5. Lesion localization procedure.
6. Image-level metadata.
7. Train/validation/test split generation.
8. JSONL conversion procedure.
9. Any image preprocessing.
10. Any resizing or normalization applied before model input.
11. Final dataset statistics.

The final manuscript-associated dataset release should use the same split definitions as the reported experiments.

---

# 13. Dataset Leakage Prevention

The dataset is divided into:

```text
Training
Validation
Test
```

The test set should remain isolated from model training and parameter optimization.

When reproducing reported evaluation results, researchers should use the designated test partition:

```text
589 radiographs
1,767 JSONL examples
```

rather than reconstructing a new test split unless a separate experiment explicitly requires it.

---

# 14. Data-to-Model Relationship

The dataset can be represented conceptually as:

```text
                    Radiographic Dataset
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
       Classification    Counting    Localization
                                         │
                                         ▼
                              Lesion bounding boxes
                                         │
                                         ▼
                                    PAI 3–5
                                         │
                                         ▼
                              Vision model pipeline
                                         │
                                         ▼
                                  ENDOLLM system
```

The dataset therefore provides structured radiographic information that can support the visual component of the broader multimodal architecture.

---

# 15. Important Distinction: Dataset vs Knowledge Base

The ENDOLLM repository contains multiple forms of research data.

They should not be treated as one dataset.

### Radiographic dataset

Contains:

* radiographs;
* classification examples;
* lesion-counting examples;
* lesion-localization examples;
* lesion bounding boxes;
* PAI 3–5 annotations.

### RAG knowledge base

Contains:

* textual knowledge chunks;
* embedding vectors;
* metadata;
* source/page information.

### Model checkpoints

Contain:

* trained neural-network parameters.

These three categories have different provenance, licensing, and redistribution considerations.

---

# 16. Current Dataset Statistics

For the manuscript-associated ENDOLLM dataset:

```text
Total images:             3,924
Total JSONL examples:    11,772

Training images:         2,747
Training examples:       8,241

Validation images:         588
Validation examples:     1,764

Test images:               589
Test examples:           1,767

Tasks per image:             3

Localization:
    Periapical lesions
    PAI 3–5
    Original-image pixel coordinates
```

---

# 17. Current Status

This dataset documentation corresponds to the dataset structure verified for the ENDOLLM project.

```text
ENDOLLM
Project stage: 7/9

Current implementation:
Phase 4.5 Interactive Endodontic MLLM (v65)

Current diagnostic-rule stage:
Stage 7.2
```

Future changes to the dataset, annotations, split assignments, or preprocessing should be versioned and documented rather than silently replacing the manuscript-associated dataset.

---

## Citation and Source Information

Formal dataset citation information will be added after the original dataset source and publication details have been verified.

The repository should cite the original dataset creators and source rather than presenting the dataset as an original ENDOLLM dataset if the underlying radiographs or annotations originated from an external source.

---

## Summary

The verified ENDOLLM radiographic dataset consists of:

**3,924 radiographs → 11,772 task examples**

distributed as:

```text
Training:    2,747 images / 8,241 examples
Validation:    588 images / 1,764 examples
Test:          589 images / 1,767 examples
```

Each image contributes classification, counting, and localization examples, with periapical-lesion localization represented using bounding boxes and PAI classes 3–5.

The dataset is one component of the larger ENDOLLM multimodal architecture and should be distinguished from the RAG knowledge base and external model components.
