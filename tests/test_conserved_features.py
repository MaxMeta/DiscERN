import json
from pathlib import Path

import discern.funcs_2 as funcs_2
from discern.funcs_2 import (
    analyse_genbank_set,
    find_conserved_features,
    find_conserved_flat_features,
    flatten_nested_dict,
)

DATA = Path(__file__).resolve().parents[1] / "discern" / "data"

NESTED_A = {"CDS": {"NRPS_PKS": {"Domain: AMP-binding": 3, "Domain: TE": 1},
                    "gene_functions": {"SMCOG1001": 2}}}
NESTED_B = {"CDS": {"NRPS_PKS": {"Domain: AMP-binding": 2},
                    "gene_functions": {"SMCOG1001": 1, "SMCOG1002": 1}}}
# make_counts' simplified form, when only one feature/qualifier combination matched
SIMPLIFIED = {"SMCOG1001": 1}


def test_flat_matches_nested_for_nested_inputs():
    expected = flatten_nested_dict(find_conserved_features([NESTED_A, NESTED_B]))
    got = find_conserved_flat_features([flatten_nested_dict(NESTED_A), flatten_nested_dict(NESTED_B)])
    assert got == expected == {"CDS.NRPS_PKS.Domain: AMP-binding": 2, "CDS.gene_functions.SMCOG1001": 1}


def test_flat_matches_nested_for_bundled_mibig_counts():
    with open(DATA / "mibig_counts_rf.json") as handle:
        counts = json.load(handle)
    family = ["BGC0000315", "BGC0000336", "BGC0000354", "BGC0000439"]
    nested = [counts[bgc] for bgc in family]
    assert flatten_nested_dict(find_conserved_features(nested)) == \
        find_conserved_flat_features([flatten_nested_dict(c) for c in nested])


def test_analyse_genbank_set_handles_simplified_counts(monkeypatch):
    fake_counts = {"a.gbk": SIMPLIFIED, "b.gbk": {"SMCOG1001": 2, "SMCOG1003": 1}}
    monkeypatch.setattr(funcs_2, "make_counts", lambda path, criteria: fake_counts[path])
    assert analyse_genbank_set(["a.gbk", "b.gbk"], {"CDS": {"gene_functions": ["SMCOG"]}}) == \
        {"SMCOG1001": 1}


def test_analyse_genbank_set_mixed_forms_do_not_crash(monkeypatch):
    fake_counts = {"a.gbk": NESTED_A, "b.gbk": SIMPLIFIED}
    monkeypatch.setattr(funcs_2, "make_counts", lambda path, criteria: fake_counts[path])
    # Keys differ between the two forms, so nothing is shared; no AttributeError.
    assert analyse_genbank_set(["a.gbk", "b.gbk"], {"CDS": {"gene_functions": ["SMCOG"]}}) == {}


def test_no_files_no_features():
    assert find_conserved_flat_features([]) == {}
