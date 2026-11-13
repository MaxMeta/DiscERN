"""
Native homology search for the blast-vec and BLAST-rank methods.

BGC proteins are searched with DIAMOND blastp against MIBiG (and custom
reference) proteins, and hits are scored the way DiscERN scores antiSMASH
KnownClusterBlast output: for each (query BGC, subject BGC) pair, the score
is the sum over query genes of their best bitscore against that subject.

Search settings and hit filters follow antiSMASH's KnownClusterBlast
(antismash/modules/clusterblast/core.py and known.py):

- diamond blastp --evalue 1e-05 --max-target-seqs 10000 (default sensitivity)
- only the first hit for each (query protein, subject protein) pair is used
- identity is rounded to an integer and must be > 45; query coverage
  (alignment length / query length * 100) must be > 40
- bitscores are rounded to integers
- a subject BGC is kept only if it was hit by at least 2 query genes, on at
  least 2 different subject genes; subject BGCs with a single gene are exempt
- each vector keeps at most `cap` subject BGCs (antiSMASH reports 50). Here
  they are ranked by score, and the cap is applied after MIBiG and custom
  hits are merged. antiSMASH ranks by a synteny-based score instead.

Protein IDs are "<bgc>|<index>"; everything before the last "|" names the BGC.
"""

import gzip
import json
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass

from Bio import SeqIO

SEP = "|"

DIAMOND_FIELDS = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qlen"]


@dataclass(frozen=True)
class HomologySettings:
    evalue: float = 1e-5
    max_target_seqs: int = 10000
    min_identity: float = 45.0   # exclusive bound, on identity rounded to an integer
    min_coverage: float = 40.0   # exclusive bound, alignment length / query length * 100
    cap: int = 50                # subject BGCs kept per vector after merging; 0 = no cap

    def as_dict(self):
        return asdict(self)


DEFAULT_SETTINGS = HomologySettings()


class DiamondError(RuntimeError):
    """DIAMOND is missing or failed."""


# --- protein FASTA files -------------------------------------------------------

def protein_bgc(protein_id):
    """'BGC0000001|3' -> 'BGC0000001'."""
    return protein_id.rsplit(SEP, 1)[0]


def read_cds_translations(gbk_path):
    """Protein sequences of the CDS features with a translation, in file order."""
    seqs = []
    for record in SeqIO.parse(gbk_path, "genbank"):
        for feature in record.features:
            if feature.type == "CDS" and "translation" in feature.qualifiers:
                seq = feature.qualifiers["translation"][0].strip()
                if seq:
                    seqs.append(seq)
    return seqs


def write_bgc_fasta(bgcs, out_path):
    """
    Write proteins as ">bgc|index".

    Args:
        bgcs: iterable of (bgc_name, [protein sequences])
    Returns:
        {bgc_name: number of proteins written}
    """
    sizes = {}
    with open(out_path, "w") as handle:
        for name, seqs in bgcs:
            if SEP in name or any(c.isspace() for c in name):
                raise ValueError(f"BGC name {name!r} must not contain '{SEP}' or whitespace")
            sizes[name] = len(seqs)
            for i, seq in enumerate(seqs):
                handle.write(f">{name}{SEP}{i}\n{seq}\n")
    return sizes


# --- DIAMOND -------------------------------------------------------------------

def find_diamond(diamond=None):
    """Return the DIAMOND executable, or raise DiamondError with install advice."""
    exe = diamond or shutil.which("diamond")
    if not exe:
        raise DiamondError(
            "the 'diamond' executable was not found on PATH. It is installed with "
            "antiSMASH; otherwise install it with 'conda install -c bioconda diamond'."
        )
    return exe


def diamond_version(diamond=None):
    exe = find_diamond(diamond)
    out = subprocess.run([exe, "version"], capture_output=True, text=True, check=True).stdout
    return out.strip().replace("diamond version ", "")


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise DiamondError(
            f"diamond failed (exit {result.returncode}): {' '.join(args)}\n{result.stderr[-2000:]}"
        )
    return result


def make_diamond_db(fasta, db_prefix, threads=1, diamond=None):
    """Build '<db_prefix>.dmnd' from a protein FASTA file; returns its path."""
    exe = find_diamond(diamond)
    _run([exe, "makedb", "--in", fasta, "--db", db_prefix, "--threads", str(max(1, threads))])
    return db_prefix + ".dmnd"


def run_blastp(query_fasta, db, out_path, threads=1, settings=DEFAULT_SETTINGS,
               tmpdir=None, diamond=None):
    """
    Run diamond blastp with antiSMASH KnownClusterBlast settings, writing a
    gzipped tabular file with DIAMOND_FIELDS. Returns the path written.
    An empty query file gives an empty result without running DIAMOND.
    """
    if not out_path.endswith(".gz"):
        out_path += ".gz"
    if os.path.getsize(query_fasta) == 0:
        with gzip.open(out_path, "wt"):
            pass
        return out_path
    exe = find_diamond(diamond)
    args = [
        exe, "blastp", "--db", db, "--query", query_fasta,
        "--out", out_path[:-3], "--compress", "1",
        "--outfmt", "6", *DIAMOND_FIELDS,
        "--evalue", str(settings.evalue),
        "--max-target-seqs", str(settings.max_target_seqs),
        "--threads", str(max(1, threads)),
        # as antiSMASH: don't fail on translations that look like nucleotides
        "--ignore-warnings",
    ]
    if tmpdir:
        args += ["--tmpdir", tmpdir]
    _run(args)
    return out_path


# --- scoring -------------------------------------------------------------------

def iter_hit_lines(path):
    """Yield split DIAMOND_FIELDS rows from a (gzipped) tabular file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= len(DIAMOND_FIELDS):
                yield parts


def aggregate_hits(rows, settings=DEFAULT_SETTINGS):
    """
    Score hits per (query BGC, subject BGC) pair.

    Args:
        rows: iterable of [qseqid, sseqid, pident, length, evalue, bitscore, qlen]
    Returns:
        {query_bgc: {subject_bgc: [score, n_query_genes, n_subject_genes]}},
        before the inclusion rule and the cap (see finalize_vectors).
    """
    best = defaultdict(dict)       # (query bgc, subject bgc) -> {query protein: best score}
    subjects = defaultdict(set)    # (query bgc, subject bgc) -> subject proteins hit
    current_query = None
    seen = set()
    for qid, sid, pident, length, _evalue, bitscore, qlen, *_ in rows:
        if qid != current_query:   # DIAMOND reports all hits of a query together
            current_query = qid
            seen = set()
        if sid in seen:            # keep only the first hit per protein pair
            continue
        seen.add(sid)
        identity = int(float(pident) + 0.5)
        coverage = float(length) / float(qlen) * 100
        if identity <= settings.min_identity or coverage <= settings.min_coverage:
            continue
        score = int(float(bitscore) + 0.5)
        pair = (protein_bgc(qid), protein_bgc(sid))
        genes = best[pair]
        if score > genes.get(qid, -1):
            genes[qid] = score
        subjects[pair].add(sid)

    raw = defaultdict(dict)
    for (query, subject), genes in best.items():
        raw[query][subject] = [sum(genes.values()), len(genes), len(subjects[(query, subject)])]
    return dict(raw)


def passes_inclusion(n_query_genes, n_subject_genes, subject_size):
    """antiSMASH's rule: >= 2 query genes on >= 2 subject genes, unless the subject has one gene."""
    if subject_size == 1:
        return n_query_genes >= 1
    return n_query_genes >= 2 and n_subject_genes >= 2


def merge_raw(*raws):
    """Merge raw results (row -> subject -> stats); later arguments win on clashes."""
    merged = {}
    for raw in raws:
        for row, subjects in raw.items():
            merged.setdefault(row, {}).update(subjects)
    return merged


def finalize_vectors(raw, subject_sizes, cap=DEFAULT_SETTINGS.cap, exclude=(), rows=None):
    """
    Turn raw results into blast-vec vectors {row: {subject: score}}.

    Applies the inclusion rule, drops excluded subjects, then keeps the `cap`
    best-scoring subjects (ties broken by name; cap 0 keeps all). Vectors are
    ordered by decreasing score. `rows` adds empty vectors for rows without hits.
    """
    exclude = set(exclude)
    vectors = {}
    for row in (rows if rows is not None else raw):
        kept = [
            (subject, stats[0])
            for subject, stats in raw.get(row, {}).items()
            if subject not in exclude and passes_inclusion(stats[1], stats[2], subject_sizes.get(subject))
        ]
        kept.sort(key=lambda item: (-item[1], item[0]))
        if cap:
            kept = kept[:cap]
        vectors[row] = dict(kept)
    return vectors


def search(query_fasta, db, work_prefix, threads=1, settings=DEFAULT_SETTINGS, diamond=None):
    """Run DIAMOND and return the aggregated raw results."""
    out = run_blastp(query_fasta, db, work_prefix + ".tsv.gz", threads=threads,
                     settings=settings, tmpdir=os.path.dirname(work_prefix) or None,
                     diamond=diamond)
    return aggregate_hits(iter_hit_lines(out), settings)


# --- storage -------------------------------------------------------------------

def save_raw(raw, path, meta=None):
    with gzip.open(path, "wt") as handle:
        json.dump({"meta": meta or {}, "raw": raw}, handle)


def load_raw(path):
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    return data["raw"], data.get("meta", {})
