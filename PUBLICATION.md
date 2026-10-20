# Public snapshot and training artifacts

This repository starts from a sanitized snapshot of the development repository
at commit `4ac72fc`. The original development history is preserved on the server,
but is not included here because it contains private access/licensing material.
`ServerInfo.md`, `VideoActivationCode.txt` and `VideoBaiduYun.txt` are excluded.

Dataset migration instructions: [DATASET_SETUP.md](DATASET_SETUP.md).

The `training-artifacts-4ac72fc` GitHub Release stores the complete `outputs/`
and `weight/` directories as a split compressed archive. Download every
`training-artifacts.tar.gz.part-*` file and `SHA256SUMS` into the same directory.
Verify and extract from the project root:

```bash
shasum -a 256 -c SHA256SUMS
cat training-artifacts.tar.gz.part-* | tar -xzf -
```

Extraction recreates `outputs/` and `weight/`. Use an empty destination if you
already have training results you want to preserve. Allow at least 20 GB of
free space for extraction, in addition to downloaded archive parts.

PyTorch checkpoint files are Python serialization artifacts: load only files
from sources you trust. The upload audit scanned text files without executing
or deserializing checkpoints.
