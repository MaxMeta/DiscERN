# Bundled MIBiG data

## Used with `--cb_source antismash` (the published setup)

Built before this manifest existed; provenance to be confirmed by the maintainer.

| File | Contents |
|---|---|
| `mibig_cb_dict.json` | KnownClusterBlast blast vectors for 2,262 MIBiG BGCs; dimensions from an older KnownClusterBlast database (2,222 BGCs, all <= BGC0002749) |
| `mibig_bs_vecs.tsv` | Pfam-vec vectors for 2,260 MIBiG BGCs, from antiSMASH region GenBank files |

## Used with `--cb_source diamond`

Built 2026-11-14 by `scripts/build_mibig_reference_data.py`.

| File | Contents |
|---|---|
| `mibig_bs_vecs_4.0.json.gz` | Pfam-vec vectors for the 2636 MIBiG 4.0 BGCs, from MIBiG's protein FASTA |
| `mibig_bs_vecs_legacy.json.gz` | Pfam-vec vectors for the 244 entries dropped before 4.0 |
| `mibig_dropped_entries.json` | The dropped entries, MIBiG's retirement reasons and closest 4.0 entry (also `docs/mibig_dropped_entries.md`) |

Sources (MIBiG protein FASTA files; the DIAMOND database and blast-vec background are built from the same files on each machine by `discern-setup-mibig`):

| Release | URL | md5 |
|---|---|---|
| 4.0 | https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_4.0.fasta | `1dd73356e63d6ba3507f9199cfd22e1c` |
| 3.1 | https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_3.1.fasta | `b3cacde3f5d6b8ac27d319ce3d2677c1` |
| 3.0 | https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_3.0.fasta | `81dc4c0c45296ae22709ebea0c19202f` |
| 2.0 | https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_2.0.fasta | `c8626c785d25ac22ea5d2d447770e883` |
| 1.4 | https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_1.4.fasta | `118f2c11edde36c81bdea030a0228492` |

Metadata (names, classes, retirement reasons):

| Archive | md5 |
|---|---|
| https://dl.secondarymetabolites.org/mibig/mibig_json_4.0_all_jsons.tar.xz | `512fa0be167f9ffd142991d836692a2e` |
| https://dl.secondarymetabolites.org/mibig/mibig_json_5.0rc1.tar.bz2 | `6fac5c0887854baca0f459b3e922a3bf` |
| https://dl.secondarymetabolites.org/mibig/mibig_json_3.1.tar.gz | `643d1349722a9437d8dcf558dac5f815` |
| https://dl.secondarymetabolites.org/mibig/mibig_json_3.0.tar.gz | `7c38b90f939086c03392d99a913baef9` |
| https://dl.secondarymetabolites.org/mibig/mibig_json_2.0.tar.gz | `843ce4677db6d11422f0e6d94dd03e81` |
| https://dl.secondarymetabolites.org/mibig/mibig_json_1.4.tar.gz | `a85530571d9dd7978b1bb0f2580cd30e` |

Tools: DIAMOND 2.1.13 (closest-entry scores), BiG-SLiCE models 2022-11-30 (md5 of the archive `aaabde911ec107d08e5c24f68aaf31d1`), pyhmmer 0.12.3.
