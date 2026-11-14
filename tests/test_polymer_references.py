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


# --- ties -----------------------------------------------------------------------------

from discern.polymer_tools import best_polymer_match  # noqa: E402

TIES = {
    "R1": {"1_Ser", "1_Thr", "1_Ser - Thr"},     # references
    "R2": {"1_Ser", "1_Thr", "1_Orn"},
    "N1": {"1_Ser", "1_Gly"},                    # non-references
    "N2": {"1_Pro"},                             # shares nothing with the others
    "X": {"1_Ser", "1_Ala"},                     # ties R1, R2 and N1 at score 1
}


def test_tie_between_reference_and_non_reference_goes_outside_the_family():
    assert best_polymer_match(TIES["X"], TIES, {"R1", "R2"}, skip="X") == (1, "N1")


def test_ties_do_not_depend_on_dictionary_order():
    reordered = dict(reversed(list(TIES.items())))
    for name in TIES:
        assert best_polymer_match(TIES[name], TIES, {"R1", "R2"}, skip=name) == \
            best_polymer_match(TIES[name], reordered, {"R1", "R2"}, skip=name)
    assert best_polymer_match({"1_Ser", "1_Thr"}, TIES) == (2, "R1")   # no ref_set: smallest name


def test_no_shared_kmer_means_no_match():
    assert best_polymer_match(TIES["N2"], TIES, {"R1"}, skip="N2") == (0, None)


def test_training_ignores_ties_and_zero_scores_wherever_references_sit():
    refs = {"R1", "R2"}
    stats = get_polymer_stats(TIES, refs)
    # R1 and R2 are each other's unique best match; X (tied with N1) and N2 (no
    # overlap) are not false positives, whatever the order of the dictionary
    assert stats == {"TP": [2, 2], "FP": []}
    refs_last = {k: v for k, v in TIES.items() if k not in refs}
    refs_last.update({k: TIES[k] for k in sorted(refs)})
    assert get_polymer_stats(refs_last, refs) == stats


def test_classification_settles_ties_like_training():
    matches = get_polymer_matches({"genome.json": {"c::1": {"1_Ser", "1_Val"}, "c::2": {"1_Orn", "1_Ser"}}},
                                  TIES, {"R1", "R2"})
    assert matches["genome.json::c::1"] == (1, "N1")
    assert matches["genome.json::c::2"] == (2, "R2")


# --- cutoff -------------------------------------------------------------------------------

import random  # noqa: E402

from discern.polymer_tools import calculate_f1, find_best_f1_threshold  # noqa: E402


def counts_at(cutoff, tp_scores, fp_scores):
    tp = sum(score >= cutoff for score in tp_scores)
    fp = sum(score >= cutoff for score in fp_scores)
    return tp, fp, len(tp_scores) - tp


def test_cutoff_rejects_the_score_training_rejected():
    # the best cutoff is "more than 9"; it used to be applied as >= 9, letting both 9s through
    tp_scores, fp_scores = [8, 10, 11, 12, 13], [9, 9]
    threshold, f1, precision, recall, tp, fp, fn = find_best_f1_threshold(tp_scores, fp_scores, stringent=True)
    assert threshold == 10
    assert (tp, fp, fn) == (4, 0, 1)
    assert (f1, precision, recall) == (calculate_f1(4, 0, 1), 1.0, 0.8)


def test_cda_family_before_the_tie_fix():
    # the CDA walkthrough family's training scores before polymer ties were fixed
    tp_scores = [7, 8, 10, 13, 13, 15, 15, 27, 27, 37, 37]
    fp_scores = [1] * 10 + [2] * 3 + [3, 4, 5, 5, 5, 6, 6, 7, 8, 9]
    threshold, f1, precision, recall, tp, fp, fn = find_best_f1_threshold(tp_scores, fp_scores, stringent=True)
    assert threshold == 10          # was 9, which let the false positive scoring 9 through
    assert (tp, fp, fn) == counts_at(10, tp_scores, fp_scores) == (9, 0, 2)
    assert (precision, recall) == (1.0, 9 / 11)


@pytest.mark.parametrize("stringent", [True, False])
def test_applied_cutoff_keeps_the_best_f1_and_reports_its_own_counts(stringent):
    rng = random.Random(0)
    for _ in range(500):
        tp_scores = [rng.randint(0, 30) for _ in range(rng.randint(1, 12))]
        fp_scores = [rng.randint(0, 30) for _ in range(rng.randint(0, 30))]
        threshold, f1, precision, recall, tp, fp, fn = find_best_f1_threshold(
            tp_scores, fp_scores, stringent=stringent)
        assert isinstance(threshold, int)
        if stringent:
            assert threshold >= min(tp_scores)
        assert (tp, fp, fn) == counts_at(threshold, tp_scores, fp_scores)
        assert f1 == calculate_f1(tp, fp, fn)
        best = max(calculate_f1(*counts_at(c, tp_scores, fp_scores)) for c in range(0, 32))
        assert f1 == best, (tp_scores, fp_scores, threshold)
