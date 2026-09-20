# Third-Party Components and Licensing Notes

## Purpose

This document records the principal external models, software repositories,
and datasets referenced by the ENDOLLM v65/v65.1 implementation.

ENDOLLM does **not** automatically redistribute third-party model weights or
datasets. Researchers should obtain those components from their official
sources and follow the licenses and access conditions applicable to each
component.

## Verified components

| Component | Role in ENDOLLM | Current source/license status | Redistribution in this repository |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | Vision-language model | Hugging Face model page identifies Apache-2.0 | Do not bundle weights; obtain from official source |
| `BAAI/bge-small-en-v1.5` | RAG embedding model | Hugging Face model page identifies MIT | Do not bundle weights unless separately required |
| `OPGAgent` | Tooth enumeration / panoramic visual-analysis integration | Upstream GitHub repository identifies Apache-2.0 | ENDOLLM references the external installation; do not copy the repository or its bundled weights into ENDOLLM |
| DENTEX | Possible upstream dental panoramic dataset/resource in the broader project | Upstream repository identifies CC BY-NC-SA 4.0 for the data and MIT for the repository code | Do not redistribute dataset files unless permitted by the applicable terms |

## OPGAgent dependency boundary

ENDOLLM v65/v65.1 invokes an external OPGAgent installation and references
its tooth-enumeration and TVEM components. OPGAgent itself contains multiple
backend model families and model checkpoints. Those components can have
different provenance and licensing terms.

Therefore:

1. ENDOLLM should document the exact OPGAgent commit/version used for a
   manuscript-associated experiment when that information is available.
2. ENDOLLM should not copy OPGAgent's complete model directory into this
   repository.
3. Individual downstream weights should be downloaded from their official
   sources and used according to their own terms.
4. If a future ENDOLLM release redistributes any third-party weight, its
   specific license and redistribution permission must be checked first.

## Dataset boundary

The repository should document dataset provenance, source, publication,
subset, preprocessing, and license. Dataset files should be redistributed
only when the original license and any applicable institutional/ethical
requirements permit redistribution.

Patient-identifiable or otherwise restricted clinical data must not be added
to the public repository.

## Model-weight boundary

The public repository is intentionally code/documentation focused. Model
weights and checkpoints such as `.pt`, `.pth`, `.ckpt`, `.bin`, and
`.safetensors` are excluded by `.gitignore` unless a future release has
explicitly verified redistribution rights.

## Important licensing distinction

The fact that an upstream model or repository is publicly downloadable does
not by itself establish that its weights, datasets, or derivatives can be
redistributed by ENDOLLM.

The ENDOLLM repository should therefore distinguish:

- ENDOLLM-authored source code;
- third-party source code;
- third-party model weights;
- public datasets;
- derived/processed datasets;
- retrieval indexes and embeddings.

Licensing decisions for each category should be recorded separately.

## Current verified web sources

- Qwen2.5-VL-7B-Instruct: official Hugging Face model page.
- BAAI/bge-small-en-v1.5: official Hugging Face model page.
- OPGAgent: official GitHub repository.
- DENTEX: official GitHub repository.

These sources were checked during the ENDOLLM Stage 7 reproducibility audit
on 20 September 2026.
