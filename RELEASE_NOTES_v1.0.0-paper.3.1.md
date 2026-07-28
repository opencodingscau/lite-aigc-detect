# Release v1.0.0-paper.3.1

YAML-safe `CITATION.cff` fix for Zenodo archival.

## Why

`v1.0.0-paper.3` included an unquoted description containing `temporary: paper.2`, which is invalid YAML (`mapping values are not allowed here`). Zenodo rejected the deposit.

## Fix

- Quote DOI / description strings in `CITATION.cff`
- Prefer concept DOI `10.5281/zenodo.21604044` until this release is archived
- Content otherwise identical to `v1.0.0-paper.3` (appendices D/E, ensemble script, narrative)

## After Zenodo mints a version DOI

Back-fill `doi` + the primary identifier in `CITATION.cff`, README, and `latex/main.tex` Data Availability.
