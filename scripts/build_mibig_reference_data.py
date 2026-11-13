#!/usr/bin/env python
"""
Build the MIBiG reference data that DiscERN bundles for --cb_source diamond.

Users do not need to run this: the DIAMOND database and blast-vec background
are built on each machine with `discern-setup-mibig`. This script produces
the files committed to the repository, from pinned MIBiG releases:

  pfam      Pfam-vec (BiG-SLiCE) vectors for every BGC of the 4.0_legacy set,
            computed from MIBiG's protein sequences:
              discern/data/mibig_bs_vecs_4.0.json.gz      (MIBiG 4.0 entries)
              discern/data/mibig_bs_vecs_legacy.json.gz   (entries dropped before 4.0)
  dropped   the entries dropped before MIBiG 4.0, with MIBiG's retirement
            reasons and their closest MIBiG 4.0 entry:
              discern/data/mibig_dropped_entries.json
              docs/mibig_dropped_entries.md
  manifest  discern/data/MANIFEST.md
  parity    a report comparing the rebuilt data with the bundled
            KnownClusterBlast-based files (written to --report)

Needs DIAMOND on PATH and the BiG-SLiCE models in discern/bigslice_models.

    python scripts/build_mibig_reference_data.py --db_dir /path/to/mibig_db -c 8
"""

import argparse
import datetime
import gzip
import json
import os
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from discern import homology, mibig_db  # noqa: E402
from discern.funcs_2 import make_sparse_dict_from_proteins, tsv_to_sparse_dict  # noqa: E402

DATA = REPO / "discern" / "data"
DOCS = REPO / "docs"
BIGSLICE_MODELS = REPO / "discern" / "bigslice_models"

# MIBiG metadata used for names, classes and retirement reasons: (url, md5)
MIBIG_JSON = {
    "4.0_all": (f"{mibig_db.MIBIG_DOWNLOAD}/mibig_json_4.0_all_jsons.tar.xz", "512fa0be167f9ffd142991d836692a2e"),
    "5.0rc1": (f"{mibig_db.MIBIG_DOWNLOAD}/mibig_json_5.0rc1.tar.bz2", "6fac5c0887854baca0f459b3e922a3bf"),
    "3.1": (f"{mibig_db.MIBIG_DOWNLOAD}/mibig_json_3.1.tar.gz", "643d1349722a9437d8dcf558dac5f815"),
    "3.0": (f"{mibig_db.MIBIG_DOWNLOAD}/mibig_json_3.0.tar.gz", "7c38b90f939086c03392d99a913baef9"),
    "2.0": (f"{mibig_db.MIBIG_DOWNLOAD}/mibig_json_2.0.tar.gz", "843ce4677db6d11422f0e6d94dd03e81"),
    "1.4": (f"{mibig_db.MIBIG_DOWNLOAD}/mibig_json_1.4.tar.gz", "a85530571d9dd7978b1bb0f2580cd30e"),
}

NEAR_DUPLICATE_RATIO = 0.9


# --- helpers -------------------------------------------------------------------------

def fetch(url, md5, download_dir):
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / os.path.basename(url)
    if not (path.exists() and mibig_db.md5sum(path) == md5):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
        found = mibig_db.md5sum(path)
        if found != md5:
            raise SystemExit(f"md5 mismatch for {url}: expected {md5}, got {found}")
    return path


def read_json_archive(path):
    """{accession: entry} from a MIBiG JSON archive, for the old and the 4.x schema."""
    entries = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            try:
                entry = json.load(archive.extractfile(member))
            except ValueError:
                continue
            cluster = entry.get("cluster", entry.get("general_params", {}))
            accession = entry.get("accession") or cluster.get("mibig_accession") \
                or os.path.basename(member.name)[:-5]
            entries[mibig_db.unversioned(accession)] = entry
    return entries


def compounds_and_classes(entry):
    """Compound names and biosynthetic classes from either JSON schema."""
    if "cluster" in entry or "general_params" in entry:
        cluster = entry.get("cluster", entry.get("general_params", {}))
        names = [c.get("compound") for c in cluster.get("compounds", []) if c.get("compound")]
        classes = cluster.get("biosyn_class", [])
    else:
        names = [c.get("name") for c in entry.get("compounds", []) if c.get("name")]
        classes = [c.get("class") for c in entry.get("biosynthesis", {}).get("classes", []) if c.get("class")]
    return names, classes


def write_gz_json(obj, path):
    with gzip.open(path, "wt") as handle:
        json.dump(obj, handle, sort_keys=True, separators=(",", ":"))


def cosine(a, b):
    keys = sorted(set(a) | set(b))
    x = np.array([a.get(k, 0) for k in keys], dtype=float)
    y = np.array([b.get(k, 0) for k in keys], dtype=float)
    if not x.any() or not y.any():
        return float("nan")
    return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))


# --- steps -----------------------------------------------------------------------------

def read_gz_json(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def step_pfam(legacy, threads, recompute=False):
    """Pfam-vec vectors; existing bundled vectors are kept unless recompute is set."""
    if not BIGSLICE_MODELS.exists():
        raise SystemExit(f"BiG-SLiCE models not found in {BIGSLICE_MODELS}")
    proteins = legacy.proteins_by_bgc()
    vectors = {}
    if not recompute:
        for name in ("mibig_bs_vecs_4.0.json.gz", "mibig_bs_vecs_legacy.json.gz"):
            if (DATA / name).exists():
                vectors.update({k: v for k, v in read_gz_json(DATA / name).items() if k in proteins})
        if vectors:
            print(f"pfam-vec: keeping {len(vectors)} existing vectors (--recompute_pfam to redo)")
    missing = {k: v for k, v in proteins.items() if k not in vectors}
    if missing:
        vectors.update(make_sparse_dict_from_proteins(missing, str(BIGSLICE_MODELS), num_cpus=threads))
    current = {k: v for k, v in vectors.items() if legacy.origin[k] == "4.0"}
    dropped = {k: v for k, v in vectors.items() if legacy.origin[k] != "4.0"}
    write_gz_json(current, DATA / "mibig_bs_vecs_4.0.json.gz")
    write_gz_json(dropped, DATA / "mibig_bs_vecs_legacy.json.gz")
    print(f"pfam-vec: {len(current)} MIBiG 4.0 + {len(dropped)} dropped entries "
          f"({sum(1 for v in vectors.values() if not v)} without biosynthetic Pfam hits)")


def closest_current_entry(legacy, bgc):
    """(closest MIBiG 4.0 entry, score / self score) from the 4.0_legacy background."""
    row = legacy.background_raw().get(bgc, {})
    self_score = row.get(bgc, [0])[0]
    others = [(s, stats[0]) for s, stats in row.items() if s != bgc and legacy.origin.get(s) == "4.0"]
    if not others or not self_score:
        return None, 0.0
    best, score = max(others, key=lambda item: (item[1], item[0]))
    return best, round(score / self_score, 3)


def describe_reason(bgc, last_release, meta40, meta50):
    entry = meta40.get(bgc)
    later = meta50.get(bgc, {})
    notes = []
    if entry and entry.get("status") == "retired":
        reasons = entry.get("retirement_reasons") or ["no reason given"]
        text = "; ".join(reasons)
        if entry.get("see_also"):
            text += f" (see also {', '.join(entry['see_also'])})"
        notes.append(text)
    elif entry:
        notes.append(f"listed as '{entry.get('status')}' in the MIBiG 4.0 metadata, "
                     "but missing from the MIBiG 4.0 protein FASTA")
    elif last_release == "3.1":
        notes.append("not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata "
                     "(it was 'active' in 3.1)")
    else:
        notes.append(f"not in MIBiG releases after {last_release}; no retirement record "
                     "in the MIBiG 4.0 metadata")
    if later.get("status") == "active":
        notes.append("listed as active again in MIBiG 5.0rc1")
    return "; ".join(notes)


def step_dropped(legacy, download_dir):
    meta = {key: read_json_archive(fetch(url, md5, download_dir)) for key, (url, md5) in MIBIG_JSON.items()}
    releases = ["3.1", "3.0", "2.0", "1.4"]
    entries = {}
    for bgc in sorted(legacy.legacy_ids()):
        last = legacy.origin[bgc]
        source = next((meta[r][bgc] for r in releases if bgc in meta[r]), None) \
            or meta["4.0_all"].get(bgc, {})
        names, classes = compounds_and_classes(source)
        closest, ratio = closest_current_entry(legacy, bgc)
        entries[bgc] = {
            "last_release": last,
            "compounds": names,
            "classes": classes,
            "reason": describe_reason(bgc, last, meta["4.0_all"], meta["5.0rc1"]),
            "closest_4.0_entry": closest,
            "closest_score_ratio": ratio,
            "near_duplicate": bool(closest) and ratio >= NEAR_DUPLICATE_RATIO,
        }
    with open(DATA / "mibig_dropped_entries.json", "w") as handle:
        json.dump(entries, handle, indent=1, sort_keys=True)

    by_release = Counter(e["last_release"] for e in entries.values())
    near = sorted(b for b, e in entries.items() if e["near_duplicate"])
    lines = [
        "# MIBiG entries dropped before MIBiG 4.0",
        "",
        "DiscERN's native homology source (`--cb_source diamond`) uses MIBiG 4.0 as published "
        "by default. These entries were in earlier MIBiG releases but are not in 4.0. They can be "
        "used as references, and are part of the background, with `--mibig_set 4.0_legacy` "
        "(build it once with `discern-setup-mibig --set 4.0_legacy`). Each one is taken from "
        "the most recent release that still contained it.",
        "",
        f"{len(entries)} entries; last seen in "
        + ", ".join(f"{r} ({by_release[r]})" for r in releases if by_release[r]) + ".",
        "",
        "**Reason dropped** is MIBiG's retirement reason from the MIBiG 4.0 metadata "
        "(`mibig_json_4.0_all_jsons`), where there is one.",
        "",
        "**Closest 4.0 entry** is the MIBiG 4.0 BGC with the highest DIAMOND blast score against "
        "the dropped entry, and the **ratio** is that score divided by the entry's score against "
        f"itself. Entries with a ratio of {NEAR_DUPLICATE_RATIO} or more are marked as near-duplicates "
        f"({len(near)} entries): in the 4.0_legacy set they sit in the background as near-identical "
        "copies of a current entry, which tightens the blast-vec threshold and can take the top "
        "BLAST hit of a query away from that entry. If the current entry is in your reference family, "
        "add the near-duplicate to the references or exclude it with `-x`.",
        "",
        "| ID | Compound(s) | Class | Last release | Reason dropped | Closest 4.0 entry (ratio) |",
        "|---|---|---|---|---|---|",
    ]
    for bgc, e in entries.items():
        closest = f"{e['closest_4.0_entry']} ({e['closest_score_ratio']:.2f})" if e["closest_4.0_entry"] else "–"
        if e["near_duplicate"]:
            closest = f"**{closest}** near-duplicate"
        names = e["compounds"][:3] + ([f"+{len(e['compounds']) - 3} more"] if len(e["compounds"]) > 3 else [])
        cells = [bgc, ", ".join(names) or "–", ", ".join(e["classes"]) or "–",
                 e["last_release"], e["reason"], closest]
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "mibig_dropped_entries.md").write_text("\n".join(lines) + "\n")
    print(f"dropped entries: {len(entries)} ({len(near)} near-duplicates)")


def step_manifest(legacy):
    diamond = homology.diamond_version()
    lines = [
        "# Bundled MIBiG data",
        "",
        "## Used with `--cb_source antismash` (the published setup)",
        "",
        "Built before this manifest existed; provenance to be confirmed by the maintainer.",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `mibig_cb_dict.json` | KnownClusterBlast blast vectors for 2,262 MIBiG BGCs; dimensions from an older "
        "KnownClusterBlast database (2,222 BGCs, all <= BGC0002749) |",
        "| `mibig_bs_vecs.tsv` | Pfam-vec vectors for 2,260 MIBiG BGCs, from antiSMASH region GenBank files |",
        "",
        "## Used with `--cb_source diamond`",
        "",
        f"Built {datetime.date.today().isoformat()} by `scripts/build_mibig_reference_data.py`.",
        "",
        "| File | Contents |",
        "|---|---|",
        f"| `mibig_bs_vecs_4.0.json.gz` | Pfam-vec vectors for the {sum(1 for r in legacy.origin.values() if r == '4.0')} "
        "MIBiG 4.0 BGCs, from MIBiG's protein FASTA |",
        f"| `mibig_bs_vecs_legacy.json.gz` | Pfam-vec vectors for the {len(legacy.legacy_ids())} entries dropped before 4.0 |",
        "| `mibig_dropped_entries.json` | The dropped entries, MIBiG's retirement reasons and closest 4.0 entry "
        "(also `docs/mibig_dropped_entries.md`) |",
        "",
        "Sources (MIBiG protein FASTA files; the DIAMOND database and blast-vec background are built "
        "from the same files on each machine by `discern-setup-mibig`):",
        "",
        "| Release | URL | md5 |",
        "|---|---|---|",
    ]
    for release, (url, md5) in mibig_db.RELEASE_FASTAS.items():
        lines.append(f"| {release} | {url} | `{md5}` |")
    lines += [
        "",
        "Metadata (names, classes, retirement reasons):",
        "",
        "| Archive | md5 |",
        "|---|---|",
    ]
    for key, (url, md5) in MIBIG_JSON.items():
        lines.append(f"| {url} | `{md5}` |")
    lines += [
        "",
        f"Tools: DIAMOND {diamond} (closest-entry scores), BiG-SLiCE models 2022-11-30 "
        "(md5 of the archive `aaabde911ec107d08e5c24f68aaf31d1`), "
        f"pyhmmer {__import__('pyhmmer').__version__}.",
    ]
    (DATA / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def step_parity(current, report_path, cap):
    """Compare the rebuilt MIBiG 4.0 data with the bundled KnownClusterBlast-based files."""
    with open(DATA / "mibig_cb_dict.json") as handle:
        old = json.load(handle)
    old_dims = {d for v in old.values() for d in v}
    new = homology.finalize_vectors(current.background_raw(), current.sizes, cap=cap)
    limit = "BGC0002749"
    dims = {d for d in old_dims | set(new) if d <= limit and d in old_dims and d in current.ids}
    rows = sorted(set(old) & set(new))

    def restrict(vec):
        return {k: v for k, v in vec.items() if k in dims}

    def ranked(vec, drop=None):
        return [k for k, _ in sorted(vec.items(), key=lambda kv: (-kv[1], kv[0])) if k != drop]

    stats = []
    for row in rows:
        a, b = restrict(old[row]), restrict(new[row])
        keys = sorted(set(a) | set(b))
        rho = spearmanr([a.get(k, 0) for k in keys], [b.get(k, 0) for k in keys])[0] if len(keys) > 2 else float("nan")
        ta, tb = ranked(a), ranked(b)
        jac = len(set(ta[:10]) & set(tb[:10])) / len(set(ta[:10]) | set(tb[:10])) if (ta or tb) else float("nan")
        na, nb = ranked(a, drop=row), ranked(b, drop=row)
        stats.append({
            "row": row, "rho": rho, "jaccard10": jac,
            "same_top": bool(ta and tb and ta[0] == tb[0]),
            "same_top_nonself": bool(na and nb and na[0] == nb[0]),
            "has_nonself": bool(na or nb),
            "old_top": ta[:5], "new_top": tb[:5],
        })

    rhos = np.array([s["rho"] for s in stats if not np.isnan(s["rho"])])
    jacs = np.array([s["jaccard10"] for s in stats if not np.isnan(s["jaccard10"])])
    nonself = [s for s in stats if s["has_nonself"]]
    old_bs = {k.split("/")[-1].split(".")[0]: v for k, v in tsv_to_sparse_dict(str(DATA / "mibig_bs_vecs.tsv")).items()}
    new_bs = read_gz_json(DATA / "mibig_bs_vecs_4.0.json.gz")
    cos = np.array([cosine(old_bs[k], new_bs[k]) for k in sorted(set(old_bs) & set(new_bs))])
    cos = cos[~np.isnan(cos)]

    worst = sorted([s for s in stats if not np.isnan(s["rho"])], key=lambda s: s["rho"])[:8]
    lines = [
        "# Phase 1 parity: rebuilt MIBiG 4.0 background vs bundled data",
        "",
        f"Rows compared: {len(rows)} (in both the bundled `mibig_cb_dict.json` and MIBiG 4.0). "
        f"Dimensions restricted to the {len(dims)} BGCs <= {limit} present in both. "
        f"Rebuilt vectors capped at {cap} (bundled: KnownClusterBlast top 50).",
        "",
        "## blast-vec background",
        "",
        f"- Spearman correlation per row: median {np.median(rhos):.3f}, mean {rhos.mean():.3f}, "
        f"10th percentile {np.percentile(rhos, 10):.3f} (n={len(rhos)}; rows with <3 dimensions skipped)",
        f"- Jaccard overlap of top-10 lists: median {np.median(jacs):.3f}, mean {jacs.mean():.3f}",
        f"- Same top hit (self included): {100 * np.mean([s['same_top'] for s in stats]):.1f}% of rows",
        f"- Same top non-self hit: {100 * np.mean([s['same_top_nonself'] for s in nonself]):.1f}% "
        f"of the {len(nonself)} rows with a non-self hit",
        "",
        "Biggest disagreements (lowest Spearman):",
        "",
        "| Row | rho | bundled top 5 | rebuilt top 5 |",
        "|---|---|---|---|",
    ]
    for s in worst:
        lines.append(f"| {s['row']} | {s['rho']:.2f} | {' '.join(s['old_top'])} | {' '.join(s['new_top'])} |")
    lines += [
        "",
        "## Pfam-vec",
        "",
        f"Cosine similarity between bundled (antiSMASH region GenBank) and rebuilt (MIBiG protein FASTA) "
        f"vectors for {len(cos)} shared BGCs: median {np.median(cos):.3f}, mean {cos.mean():.3f}, "
        f"{100 * np.mean(cos > 0.999):.1f}% > 0.999, {100 * np.mean(cos < 0.9):.1f}% < 0.9. "
        f"Bundled rows: {len(old_bs)}; rebuilt MIBiG 4.0 rows: {len(new_bs)} "
        f"({sum(1 for v in new_bs.values() if not v)} without biosynthetic Pfam hits).",
    ]
    Path(report_path).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db_dir", default=None, help="where the MIBiG sets are (or will be) built")
    parser.add_argument("--download_dir", default=None, help="where to keep downloads")
    parser.add_argument("-c", "--num_cpus", type=int, default=0)
    parser.add_argument("--steps", default="sets,pfam,dropped,manifest,parity")
    parser.add_argument("--report", default="parity_report.md")
    parser.add_argument("--cap", type=int, default=homology.DEFAULT_SETTINGS.cap)
    parser.add_argument("--recompute_pfam", action="store_true",
                        help="recompute all Pfam-vec vectors instead of only missing ones")
    args = parser.parse_args()
    steps = set(args.steps.split(","))
    root = Path(args.db_dir or mibig_db.default_db_root())
    download_dir = Path(args.download_dir or root / "downloads")

    if "sets" in steps:
        for name in mibig_db.MIBIG_SETS:
            mibig_db.build_set(name, root=root, threads=args.num_cpus, download_dir=download_dir)
    legacy = mibig_db.load_set("4.0_legacy", root)
    if "pfam" in steps:
        step_pfam(legacy, args.num_cpus, recompute=args.recompute_pfam)
    if "dropped" in steps:
        step_dropped(legacy, download_dir)
    if "manifest" in steps:
        step_manifest(legacy)
    if "parity" in steps:
        step_parity(mibig_db.load_set("4.0", root), args.report, args.cap)


if __name__ == "__main__":
    main()
