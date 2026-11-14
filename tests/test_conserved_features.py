import json
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

import discern.funcs_2 as funcs_2
from discern.funcs_2 import (
    analyse_genbank_set,
    analyse_mibig_set,
    conserved_features_from_counts,
    filter_by_domains,
    find_conserved_features,
    find_conserved_flat_features,
    flatten_nested_dict,
    make_counts,
    nest_feature_counts,
    qualify_feature_keys,
)

DATA = Path(__file__).resolve().parents[1] / "discern" / "data"

FEATURES = {"CDS": {"NRPS_PKS": ["Domain: AMP-binding", "Domain: TE"],
                    "gene_functions": ["SMCOG1001", "SMCOG1002", "SMCOG1003"]}}
NESTED_A = {"CDS": {"NRPS_PKS": {"Domain: AMP-binding": 3, "Domain: TE": 1},
                    "gene_functions": {"SMCOG1001": 2}}}
NESTED_B = {"CDS": {"NRPS_PKS": {"Domain: AMP-binding": 2},
                    "gene_functions": {"SMCOG1001": 1, "SMCOG1002": 1}}}
# older make_counts' simplified form, when only one feature/qualifier combination matched
SIMPLIFIED = {"SMCOG1001": 1}


def bundled(name):
    with open(DATA / name) as handle:
        return json.load(handle)


# --- normalising the older simplified form -----------------------------------------

def test_nest_feature_counts_restores_the_qualifier():
    assert nest_feature_counts(SIMPLIFIED, FEATURES) == {"CDS": {"gene_functions": {"SMCOG1001": 1}}}
    assert nest_feature_counts({"Domain: TE": 2}, FEATURES) == {"CDS": {"NRPS_PKS": {"Domain: TE": 2}}}
    assert nest_feature_counts(NESTED_A, FEATURES) is NESTED_A
    assert nest_feature_counts({}, FEATURES) == {}


def test_qualify_feature_keys():
    flat = {"SMCOG1001": 1, "CDS.NRPS_PKS.Domain: TE": 2}
    assert qualify_feature_keys(flat, FEATURES) == {"CDS.gene_functions.SMCOG1001": 1, "CDS.NRPS_PKS.Domain: TE": 2}


def test_every_bundled_count_can_be_nested():
    features, counts = bundled("feature_dict.json"), bundled("mibig_counts_rf.json")
    nested = {k: nest_feature_counts(v, features) for k, v in counts.items()}
    assert all(all(isinstance(q, dict) for f in v.values() for q in f.values()) for v in nested.values())


def test_make_counts_no_longer_simplifies(tmp_path):
    record = SeqRecord(Seq("A" * 300), id="r", annotations={"molecule_type": "DNA"})
    record.features.append(SeqFeature(FeatureLocation(0, 300, strand=1), type="CDS",
                                      qualifiers={"gene_functions": ["biosynthetic (rule-based-clusters) SMCOG1001: x"]}))
    path = tmp_path / "r.gbk"
    SeqIO.write(record, str(path), "genbank")
    assert make_counts(str(path), FEATURES) == {"CDS": {"gene_functions": {"SMCOG1001": 1}}}


# --- conserved features ---------------------------------------------------------------

def test_flat_matches_nested_for_nested_inputs():
    expected = flatten_nested_dict(find_conserved_features([NESTED_A, NESTED_B]))
    got = find_conserved_flat_features([flatten_nested_dict(NESTED_A), flatten_nested_dict(NESTED_B)])
    assert got == expected == {"CDS.NRPS_PKS.Domain: AMP-binding": 2, "CDS.gene_functions.SMCOG1001": 1}


def test_unchanged_for_bundled_families_that_worked_before():
    counts, features = bundled("mibig_counts_rf.json"), bundled("feature_dict.json")
    for family in (["BGC0000315", "BGC0000336", "BGC0000354", "BGC0000439"],
                   "BGC0000289 BGC0000290 BGC0000311 BGC0000322 BGC0000326 BGC0000418".split()):
        old = flatten_nested_dict(find_conserved_features([counts[b] for b in family]))
        assert analyse_mibig_set(family, counts, features) == old


def test_families_with_simplified_entries_no_longer_crash():
    counts, features = bundled("mibig_counts_rf.json"), bundled("feature_dict.json")
    assert counts["BGC0000500"] == {"SMCOG1070": 1, "SMCOG1288": 1, "SMCOG1075": 1}   # simplified
    assert analyse_mibig_set(["BGC0000500", "BGC0000501"], counts, features) == \
        {"CDS.gene_functions.SMCOG1288": 1}


def test_a_reference_lacking_a_feature_vetoes_it():
    # the simplified entry has no NRPS/PKS domains, so none are conserved
    features, empty = conserved_features_from_counts([("a", NESTED_A), ("b", SIMPLIFIED)], FEATURES)
    assert features == {"CDS.gene_functions.SMCOG1001": 1}
    assert empty == []


def test_references_without_counts_are_left_out():
    features, empty = conserved_features_from_counts([("a", NESTED_A), ("b", NESTED_B), ("c", {})], FEATURES)
    assert features == {"CDS.NRPS_PKS.Domain: AMP-binding": 2, "CDS.gene_functions.SMCOG1001": 1}
    assert empty == ["c"]


def test_analyse_genbank_set_handles_simplified_counts(monkeypatch):
    fake_counts = {"a.gbk": SIMPLIFIED, "b.gbk": {"SMCOG1001": 2, "SMCOG1003": 1}}
    monkeypatch.setattr(funcs_2, "make_counts", lambda path, criteria: fake_counts[path])
    assert analyse_genbank_set(["a.gbk", "b.gbk"], FEATURES) == {"CDS.gene_functions.SMCOG1001": 1}


def test_no_files_no_features():
    assert find_conserved_flat_features([]) == {}


# --- filtering queries --------------------------------------------------------------------

def test_query_with_simplified_counts_now_passes_the_filter():
    conserved = {"CDS.gene_functions.SMCOG1288": 1}
    old_style = {"q.gbk": {"SMCOG1288": 1, "SMCOG1070": 2}}     # saved by an older run
    assert filter_by_domains(["q.gbk"], old_style, conserved) == []
    fixed = {k: qualify_feature_keys(v, {"CDS": {"gene_functions": ["SMCOG1288", "SMCOG1070"]}})
             for k, v in old_style.items()}
    assert filter_by_domains(["q.gbk"], fixed, conserved) == ["q.gbk"]
