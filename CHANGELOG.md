# Changelog

## Unreleased (branch `fix/custom-reference-dir`)

### Changes that affect results with MIBiG references

These fixes change some results of the default setup (`--cb_source antismash` with MIBiG IDs), so they differ from the published results produced with v1.0.0:

- **BLAST-rank** compared each query's top `k-1` hits with a family signature learned from the top `k`. It now checks the top `k`, so BLAST-rank finds more hits in large, tight families. On the walkthrough data, NZ_CP023699.1.region004 becomes a glycopeptide BLAST-rank hit and moves from k=2 to k=3.
- **Polymer method:** threshold training gave tied best matches to whichever BGC came last in the dictionary, and classification to whichever came first. Both now settle ties the same way and independently of order: a tie between a reference and a non-reference goes to the non-reference, and BGCs sharing no k-mer have no match. This removes false positives from training (CDA walkthrough family: 23 to 5, threshold 9 to 7); walkthrough hits are unchanged.
- **Polymer cutoff:** the trained cutoff was rounded to the nearest whole number before use, so a cutoff of "more than 9" was applied as "9 or more" and let through matches with the score training had rejected. It is now rounded up, and the precision and recall it reports describe the cutoff actually applied.
- **Conserved-domain filter:** feature counts with a single feature/qualifier combination (604 of the bundled MIBiG entries, most RiPPs and terpenes) crashed the filter, and query counts of that form were never matched. Both now work; results for reference sets that worked before are unchanged.
- KnownClusterBlast output from antiSMASH 8 names MIBiG hits with a version suffix (`BGC0000315.5`); the suffix is now dropped so it matches the bundled data.
- MIBiG references without bundled data stop the run with a clear message, and the polymer tree no longer fails on BGCs without a polymer prediction.

### Custom references (`-r <directory>`)

- Fixed the crash before any analysis, the polymer reference format, file-name checks with clear messages, and the conserved-domain filter for custom references. The run states what custom mode can't do and caps `--min_k` at the reachable k.

### New

- `--cb_source diamond`: DiscERN's own DIAMOND search against MIBiG and custom references, scored like KnownClusterBlast, so custom references take part in all four methods. Needs `discern-setup-mibig`.
- `--mibig_set 4.0` (default) or `4.0_legacy` (MIBiG 4.0 plus the 244 entries dropped before 4.0; see `docs/mibig_dropped_entries.md`).
- `--custom_refs` to mix MIBiG and custom references, `--cb_cap`, `--mibig_db_dir`, `run_info.json`.
- `scripts/build_mibig_reference_data.py` and `discern/data/MANIFEST.md` for the bundled MIBiG data; pytest tests in `tests/`.
