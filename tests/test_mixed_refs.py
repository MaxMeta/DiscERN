"""Mixed MIBiG + custom references and the diamond source, without DIAMOND."""

import pytest

import discern.funcs_2 as funcs_2
from discern.custom_refs import (
    CustomReferenceError,
    cap_min_k,
    check_reference_files,
    max_achievable_k,
    validate_reference_vectors,
)
from discern.funcs_2 import (
    conserved_features_for_references,
    find_hits_intersection,
    make_cb_lists,
    make_cb_lists_drop_self_by_id,
)
from discern.vector_tools import make_trees


def touch(directory, *names):
    for name in names:
        (directory / name).write_text("")


# --- self dropped by ID ---------------------------------------------------------

def test_self_dropped_by_id_not_position():
    vectors = {
        "A": {"A": 900, "B": 500, "C": 100},
        "B": {"A": 700, "B": 600, "D": 50},   # self is not the top hit
        "C": {"A": 300, "D": 200},            # no self dimension at all
    }
    assert make_cb_lists_drop_self_by_id(vectors, ["A", "B", "C"]) == [["B", "C"], ["A", "D"], ["A", "D"]]
    # the positional version throws away B's best hit and C's real top hit
    assert make_cb_lists(vectors, ["A", "B", "C"], skip_self=False) == [["B", "C"], ["B", "D"], ["D"]]


def test_blast_rank_signature_with_custom_ids():
    # Self dropped by ID, each reference's lists are [BGC0000009, <other reference>, ...],
    # so the signature is: 1 family member within the top hits, top hit in the family.
    refs = {
        "BGC0000001": {"BGC0000001": 900, "BGC0000009": 850, "my_ref": 800, "BGC0000008": 100},
        "my_ref": {"my_ref": 950, "BGC0000009": 900, "BGC0000001": 820, "BGC0000008": 90},
    }
    queries = {
        "hit.gbk": {"my_ref": 600, "BGC0000001": 550, "BGC0000009": 10},
        "top_not_in_family.gbk": {"BGC0000009": 700, "my_ref": 600, "BGC0000001": 500},
        "no_family.gbk": {"BGC0000009": 700, "BGC0000008": 100},
    }
    hits = find_hits_intersection(refs, queries, ["BGC0000001", "my_ref"], drop_self_by_id=True)
    assert hits == ["hit.gbk"]


def test_blast_rank_skipped_when_a_reference_has_only_itself(capsys):
    refs = {"A": {"A": 900}, "B": {"B": 800, "A": 100}}
    assert find_hits_intersection(refs, {"q": {"A": 1}}, ["A", "B"], drop_self_by_id=True) == []
    assert "BLAST-rank skipped" in capsys.readouterr().out


# --- maximum k ----------------------------------------------------------------------

@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("poly_search, max_k", [(False, 3), (True, 4)])
def test_diamond_source_keeps_blast_rank_for_custom_refs(custom, poly_search, max_k):
    assert max_achievable_k(custom, poly_search, cb_source="diamond") == max_k
    assert cap_min_k(4, custom, poly_search, cb_source="diamond") == (max_k, max_k)


# --- custom reference files with the diamond source --------------------------------

def test_txt_not_required_with_diamond(tmp_path):
    touch(tmp_path, "a.gbk", "b.gbk", "b.json")
    assert len(check_reference_files(str(tmp_path), require_txt=False)) == 2
    with pytest.raises(CustomReferenceError, match=r"no \.json for a"):
        check_reference_files(str(tmp_path), poly_search=True, require_txt=False)


def test_names_must_be_usable_as_fasta_ids(tmp_path):
    touch(tmp_path, "bad name.gbk", "bad|name2.gbk")
    with pytest.raises(CustomReferenceError, match="must not contain"):
        check_reference_files(str(tmp_path), require_txt=False)


def test_vector_validation_without_cb_vectors():
    validate_reference_vectors("refs", ["a"], bs_vecs={"a": {"x": 1}}, cb_vecs=None)


# --- conserved features for mixed references -------------------------------------------

def test_conserved_features_merge_mibig_counts_and_custom_gbks(monkeypatch):
    mibig_counts = {
        "BGC0000001": {"CDS": {"NRPS_PKS": {"Domain: AMP-binding": 4, "Domain: TE": 1}}},
        "BGC0000002": {"CDS": {"NRPS_PKS": {"Domain: AMP-binding": 2, "Domain: TE": 1}}},
    }
    custom = {"ref.gbk": {"CDS": {"NRPS_PKS": {"Domain: AMP-binding": 3}}}}
    monkeypatch.setattr(funcs_2, "make_counts", lambda path, criteria: custom[path])
    features, missing, empty = conserved_features_for_references(
        {"BGC0000001", "BGC0000002", "BGC0000999"}, mibig_counts, ["ref.gbk"],
        {"CDS": {"NRPS_PKS": ["Domain: AMP-binding", "Domain: TE"]}})
    assert features == {"CDS.NRPS_PKS.Domain: AMP-binding": 2}
    assert missing == ["BGC0000999"]
    assert empty == []


# --- trees ------------------------------------------------------------------------------

def test_polymer_tree_tolerates_references_without_polymer_data(tmp_path):
    refs = ["BGC0000001", "BGC0000002", "legacy_ref"]
    pol = {"BGC0000001": {"1_Ser", "1_Thr"}, "BGC0000002": {"1_Ser"}}   # legacy_ref missing
    make_trees(str(tmp_path), [], refs, {}, {}, {}, {}, {}, pol,
               use_pol=True, use_bs=False, use_cb=False)
    assert (tmp_path / "newick_tree.txt").exists()


# --- BLAST-rank prefix length -------------------------------------------------------

def test_blast_rank_checks_as_many_hits_as_the_signature_uses():
    # every reference's top 2 non-self hits are family members: the signature is
    # "2 family members within the top 2", which a query must be able to meet
    refs = {
        "A": {"A": 900, "B": 800, "C": 700, "X": 10},
        "B": {"B": 900, "A": 800, "C": 700, "X": 10},
        "C": {"C": 900, "A": 800, "B": 700, "X": 10},
    }
    queries = {
        "tight.gbk": {"A": 500, "B": 400, "X": 300},    # top 2 both in the family
        "loose.gbk": {"A": 500, "X": 400, "B": 300},    # only 1 in the top 2
    }
    assert find_hits_intersection(refs, queries, ["A", "B", "C"], drop_self_by_id=True) == ["tight.gbk"]
    assert find_hits_intersection(refs, queries, ["A", "B", "C"]) == ["tight.gbk"]
