"""Scoring of DIAMOND hits (discern.homology), without running DIAMOND."""

import gzip

import pytest

from discern.homology import (
    HomologySettings,
    aggregate_hits,
    finalize_vectors,
    iter_hit_lines,
    merge_raw,
    passes_inclusion,
    write_bgc_fasta,
)


def hit(q, s, pident=90.0, length=100, bitscore=200.0, qlen=100, evalue=1e-50):
    """One DIAMOND row: qseqid sseqid pident length evalue bitscore qlen."""
    return [q, s, str(pident), str(length), str(evalue), str(bitscore), str(qlen)]


def test_score_is_sum_of_best_bitscore_per_query_gene():
    rows = [
        hit("Q|0", "S|0", bitscore=100),
        hit("Q|0", "S|1", bitscore=150),   # better hit of the same query gene
        hit("Q|1", "S|2", bitscore=50),
    ]
    assert aggregate_hits(rows) == {"Q": {"S": [200, 2, 3]}}


def test_identity_bound_is_exclusive_after_rounding():
    rows = [
        hit("Q|0", "A|0", pident=45.4),   # rounds to 45 -> dropped
        hit("Q|1", "A|1", pident=45.5),   # rounds to 46 -> kept
        hit("Q|2", "B|0", pident=45.0),   # exactly 45 -> dropped
    ]
    assert aggregate_hits(rows) == {"Q": {"A": [200, 1, 1]}}


def test_coverage_bound_is_exclusive():
    rows = [
        hit("Q|0", "A|0", length=40, qlen=100),   # 40 % -> dropped
        hit("Q|1", "A|1", length=41, qlen=100),   # 41 % -> kept
    ]
    assert aggregate_hits(rows) == {"Q": {"A": [200, 1, 1]}}


def test_bitscore_rounded_to_integer():
    rows = [hit("Q|0", "A|0", bitscore=100.5), hit("Q|1", "A|1", bitscore=100.4)]
    assert aggregate_hits(rows)["Q"]["A"][0] == 101 + 100


def test_only_first_hit_per_protein_pair_counts():
    # antiSMASH keeps the first line for each (query, subject) protein pair,
    # even if a later HSP scores higher or the first fails the filters.
    rows = [hit("Q|0", "A|0", bitscore=100), hit("Q|0", "A|0", bitscore=900),
            hit("Q|1", "A|1", pident=10), hit("Q|1", "A|1", pident=99)]
    assert aggregate_hits(rows) == {"Q": {"A": [100, 1, 1]}}


def test_custom_threshold_settings():
    settings = HomologySettings(min_identity=10, min_coverage=10)
    assert aggregate_hits([hit("Q|0", "A|0", pident=20, length=20)], settings) == \
        {"Q": {"A": [200, 1, 1]}}


def test_bgc_names_may_contain_pipes_before_the_index():
    assert aggregate_hits([hit("x|y|0", "BGC0000001|3")]) == {"x|y": {"BGC0000001": [200, 1, 1]}}


# --- inclusion rule ---------------------------------------------------------

def test_two_gene_inclusion_rule():
    assert passes_inclusion(2, 2, subject_size=5)
    assert not passes_inclusion(1, 1, subject_size=5)   # one query gene
    assert not passes_inclusion(3, 1, subject_size=5)   # all on one subject gene


def test_single_gene_subject_is_exempt():
    assert passes_inclusion(1, 1, subject_size=1)


def test_finalize_applies_rule_and_orders_by_score():
    raw = {"Q": {"A": [100, 2, 2], "B": [500, 1, 1], "C": [300, 1, 1], "D": [300, 3, 2]}}
    sizes = {"A": 4, "B": 3, "C": 1, "D": 6}
    vec = finalize_vectors(raw, sizes, cap=0)["Q"]
    assert list(vec.items()) == [("C", 300), ("D", 300), ("A", 100)]   # B fails the rule


def test_cap_applied_after_merging_mibig_and_custom():
    mibig = {"Q": {f"BGC{i:07d}": [100 + i, 2, 2] for i in range(5)}}
    custom = {"Q": {"my_ref": [1000, 3, 3]}}
    sizes = {**{f"BGC{i:07d}": 5 for i in range(5)}, "my_ref": 5}
    vec = finalize_vectors(merge_raw(mibig, custom), sizes, cap=3)["Q"]
    assert list(vec) == ["my_ref", "BGC0000004", "BGC0000003"]


def test_excluded_subjects_removed_before_cap():
    raw = {"Q": {"A": [300, 2, 2], "B": [200, 2, 2], "C": [100, 2, 2]}}
    vec = finalize_vectors(raw, {"A": 3, "B": 3, "C": 3}, cap=2, exclude={"A"})["Q"]
    assert vec == {"B": 200, "C": 100}


def test_rows_without_hits_get_empty_vectors():
    assert finalize_vectors({}, {}, rows=["Q"]) == {"Q": {}}


# --- files -------------------------------------------------------------------

def test_write_bgc_fasta_and_read_hits(tmp_path):
    sizes = write_bgc_fasta([("BGC0000001", ["MKV", "MLL"]), ("ref", ["MAA"])], str(tmp_path / "p.fa"))
    assert sizes == {"BGC0000001": 2, "ref": 1}
    assert (tmp_path / "p.fa").read_text() == ">BGC0000001|0\nMKV\n>BGC0000001|1\nMLL\n>ref|0\nMAA\n"
    hits = tmp_path / "h.tsv.gz"
    with gzip.open(hits, "wt") as handle:
        handle.write("\t".join(hit("Q|0", "ref|0")) + "\n")
    assert aggregate_hits(iter_hit_lines(str(hits))) == {"Q": {"ref": [200, 1, 1]}}


def test_names_with_separator_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_bgc_fasta([("bad name", ["MKV"])], str(tmp_path / "p.fa"))
