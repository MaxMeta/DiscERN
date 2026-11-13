"""Custom polymer references must have the same shape as mibig_pol_dict.pkl."""

import pickle
from pathlib import Path

import pytest

from discern.polymer_tools import (
    get_polymer_matches,
    get_polymer_stats,
    merge_kmer_counts,
    merged_kmer_token_set,
)

MIBIG_POL_FILE = Path(__file__).resolve().parents[1] / "discern" / "data" / "mibig_pol_dict.pkl"

TWO_REGIONS = {"r1": {"Ser": 1}, "r2": {"Ser": 1, "Ser - Thr": 1}}


@pytest.fixture(scope="module")
def mibig_pol():
    with open(MIBIG_POL_FILE, "rb") as handle:
        return pickle.load(handle)


def test_merge_kmer_counts_sums_counts():
    assert merge_kmer_counts(TWO_REGIONS) == {"Ser": 2, "Ser - Thr": 1}


def test_merged_token_set_counts_repeats_across_regions():
    # A union of per-region flattened sets would give only {"1_Ser", "1_Ser - Thr"}.
    assert merged_kmer_token_set(TWO_REGIONS) == {"1_Ser", "2_Ser", "1_Ser - Thr"}


def test_merged_token_set_of_nothing_is_empty_set():
    assert merged_kmer_token_set({}) == set()


def test_mibig_pickle_values_are_flat_sets(mibig_pol):
    assert mibig_pol
    assert all(isinstance(value, set) for value in mibig_pol.values())


def test_merged_reference_intersects_mibig_entries(mibig_pol):
    ref = merged_kmer_token_set(TWO_REGIONS)
    for value in mibig_pol.values():
        assert isinstance(ref & value, set)


def test_polymer_search_accepts_custom_reference(mibig_pol):
    ref = merged_kmer_token_set(TWO_REGIONS)
    background = dict(mibig_pol)
    background["custom_ref"] = ref
    # A query identical to the custom reference matches it best.
    matches = get_polymer_matches({"genome.json": {"contig::1": set(ref)}}, background)
    score, best = matches["genome.json::contig::1"]
    assert score == len(ref)
    assert isinstance(best, str)
    # Threshold training over MIBiG + custom references runs on the mixed dict.
    small = {key: background[key] for key in list(mibig_pol)[:20]}
    small["custom_ref"] = ref
    small["custom_ref_2"] = set(ref)
    stats = get_polymer_stats(small, ["custom_ref", "custom_ref_2"])
    assert set(stats) == {"TP", "FP"}
