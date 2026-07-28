# Release v1.0.0-paper.3.1

YAML-safe `CITATION.cff` fix for Zenodo archival (paper.3 content).

## Zenodo

- **Version DOI (cite this release):** https://doi.org/10.5281/zenodo.21643045
- **Concept DOI:** https://doi.org/10.5281/zenodo.21604044

## Why

`v1.0.0-paper.3` included an unquoted description containing `temporary: paper.2`, which is invalid YAML and blocked Zenodo archival.

## Fix

- Quote DOI / description strings in `CITATION.cff`
- Content otherwise identical to `v1.0.0-paper.3` (appendices D/E, ensemble script, narrative)