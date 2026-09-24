# Storage cleanup on 2026-09-21

The completed research conclusions are retained in README.md, STATUS.md, docs, and lightweight artifact summaries. The user confirmed that completed experiments do not need to remain reproducible.

Before cleanup:

- Project size: about 97 GB
- Artifacts size: about 96 GB
- Artifact directories: 205
- Artifact files: 219,619
- Project virtual environment: about 975 MB

Removed categories:

- Full forecast path arrays and training tensors
- Model checkpoints and model-weight copies
- Parquet inputs and detailed JSONL ledgers
- Raw responses, downloaded bodies, archive bundles, and restored copies
- Embedded binary dependencies and Python bytecode
- Files larger than 1 MiB inside artifacts, including detailed replay material
- Project and embedded experiment virtual environments

Retained categories:

- Source code, scripts, tests, and configurations
- README.md, STATUS.md, and docs
- Top-level data directory
- Lightweight reports, conclusions, protocols, completion records, verification summaries, and small metrics files

Historical configurations may still name removed artifact paths. They remain as documentation but cannot resume those experiments without regenerating their inputs.
