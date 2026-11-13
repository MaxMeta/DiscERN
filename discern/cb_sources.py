"""
Sources of the BLAST-derived vectors used by blast-vec and BLAST-rank.

- "antismash": parse antiSMASH KnownClusterBlast text files (needs antiSMASH
  run with --cb-knownclusters) against the bundled MIBiG vectors. This is the
  setup used for the published results.
- "diamond": DiscERN's own DIAMOND search of the query regions (and any custom
  references) against a MIBiG protein set built with discern-setup-mibig.
  Custom references become subjects too, so they get blast-vec dimensions and
  can appear in BLAST-rank hit lists.
"""

import glob
import gzip
import json
import os
import shutil
import tempfile

from . import homology
from .custom_refs import CustomReferenceError

CB_SOURCES = ("antismash", "diamond")

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIBIG_BS_FILES = {
    "4.0": ["mibig_bs_vecs_4.0.json.gz"],
    "4.0_legacy": ["mibig_bs_vecs_4.0.json.gz", "mibig_bs_vecs_legacy.json.gz"],
}
DROPPED_ENTRIES_FILE = os.path.join(_DATA_DIR, "mibig_dropped_entries.json")


def query_region_gbks(antismash_dir):
    """Region GenBank files in an antiSMASH output collection (one folder per genome)."""
    return sorted(glob.glob(os.path.join(antismash_dir, "*", "*.region*.gbk")))


def load_mibig_bs_vectors(set_name):
    """Pfam-vec vectors rebuilt from MIBiG protein sequences, {BGC ID: {feature: value}}."""
    vectors = {}
    for name in MIBIG_BS_FILES[set_name]:
        with gzip.open(os.path.join(_DATA_DIR, name), "rt") as handle:
            vectors.update(json.load(handle))
    return vectors


def load_dropped_entries():
    """{BGC ID: {...}} for MIBiG entries dropped before 4.0 (empty if the file is missing)."""
    if not os.path.exists(DROPPED_ENTRIES_FILE):
        return {}
    with open(DROPPED_ENTRIES_FILE) as handle:
        return json.load(handle)


def _write_query_fasta(regions, path):
    """Proteins of each region as '>q<index>|<cds index>'; returns {query id: region path}."""
    ids = {}

    def records():
        for i, region in enumerate(regions):
            ids[f"q{i}"] = region
            yield f"q{i}", homology.read_cds_translations(region)

    homology.write_bgc_fasta(records(), path)
    return ids


def diamond_cb_vectors(antismash_dir, mibig_set, output_dir, threads=0, custom_refs=None,
                       reuse_dir=None, settings=homology.DEFAULT_SETTINGS, exclude=()):
    """
    Build blast vectors with DIAMOND.

    Args:
        antismash_dir: antiSMASH output collection with the query regions
        mibig_set: a loaded mibig_db.MibigSet
        output_dir: run output folder; query hits against MIBiG are saved here
                    so later runs can reuse them with -u
        custom_refs: {label: region .gbk path} of custom references, or None
        reuse_dir: previous output folder to take query-vs-MIBiG hits from
        exclude: BGC IDs to remove from every vector, as rows and as dimensions

    Returns:
        (query_vectors {region gbk path: {subject: score}},
         reference_vectors {MIBiG ID or custom label: {subject: score}})
    """
    threads = threads or os.cpu_count() or 1
    custom_refs = dict(custom_refs or {})
    exclude = set(exclude)
    regions = query_region_gbks(antismash_dir)
    raw_name = f"cb_raw_mibig-{mibig_set.name}.json.gz"

    clash = sorted(set(custom_refs) & mibig_set.ids)
    if clash:
        raise CustomReferenceError(
            f"custom reference names clash with MIBiG IDs in the '{mibig_set.name}' set: "
            f"{', '.join(clash)}. Rename these files."
        )

    query_raw = None
    if reuse_dir and os.path.exists(os.path.join(reuse_dir, raw_name)):
        print(f"loading previously computed DIAMOND hits from {os.path.join(reuse_dir, raw_name)}")
        query_raw, _ = homology.load_raw(os.path.join(reuse_dir, raw_name))

    os.makedirs(output_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="homology_", dir=output_dir)
    try:
        query_fasta = os.path.join(work, "queries.fasta")
        query_ids = None
        if query_raw is None or custom_refs:
            print(f"extracting proteins from {len(regions)} query regions")
            query_ids = _write_query_fasta(regions, query_fasta)

        def by_path(raw):
            return {query_ids[q]: subjects for q, subjects in raw.items()}

        if query_raw is None:
            print(f"searching query proteins against MIBiG {mibig_set.name} with DIAMOND")
            query_raw = by_path(homology.search(query_fasta, mibig_set.db,
                                                os.path.join(work, "q_vs_mibig"),
                                                threads, settings))
            homology.save_raw(query_raw, os.path.join(output_dir, raw_name), {
                "mibig_set": mibig_set.name,
                "settings": settings.as_dict(),
                "diamond_version": homology.diamond_version(),
            })

        background_raw = mibig_set.background_raw()
        sizes = dict(mibig_set.sizes)

        if custom_refs:
            print(f"searching against {len(custom_refs)} custom references with DIAMOND")
            custom_fasta = os.path.join(work, "custom.fasta")
            custom_sizes = homology.write_bgc_fasta(
                ((label, homology.read_cds_translations(path)) for label, path in custom_refs.items()),
                custom_fasta,
            )
            empty = sorted(label for label, n in custom_sizes.items() if n == 0)
            if empty:
                raise CustomReferenceError(
                    "no CDS translations found in " + ", ".join(f"{label}.gbk" for label in empty)
                )
            sizes.update(custom_sizes)
            custom_db = homology.make_diamond_db(custom_fasta, os.path.join(work, "custom"), threads)

            def run(query, db, name):
                return homology.search(query, db, os.path.join(work, name), threads, settings)

            # queries get custom dimensions
            query_raw = homology.merge_raw(query_raw, by_path(run(query_fasta, custom_db, "q_vs_custom")))
            # MIBiG rows get custom dimensions, so threshold training sees no false zeros
            mibig_vs_custom = run(mibig_set.fasta, custom_db, "mibig_vs_custom")
            # custom rows against MIBiG and against each other (including self)
            custom_vs_mibig = run(custom_fasta, mibig_set.db, "custom_vs_mibig")
            custom_vs_custom = run(custom_fasta, custom_db, "custom_vs_custom")
            background_raw = homology.merge_raw(background_raw, mibig_vs_custom,
                                                custom_vs_mibig, custom_vs_custom)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    query_vectors = homology.finalize_vectors(query_raw, sizes, settings.cap, exclude, rows=regions)
    rows = [row for row in sorted(set(mibig_set.ids) | set(custom_refs)) if row not in exclude]
    reference_vectors = homology.finalize_vectors(background_raw, sizes, settings.cap, exclude, rows=rows)
    return query_vectors, reference_vectors
