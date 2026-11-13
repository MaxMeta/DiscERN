"""Loading and validating a custom reference directory (-r <directory>)."""

import pytest

import discern.custom_refs as custom_refs
from discern.custom_refs import (
    CustomReferenceError,
    cap_min_k,
    check_reference_files,
    load_custom_references,
    max_achievable_k,
    validate_reference_vectors,
)


def touch(directory, *names):
    for name in names:
        (directory / name).write_text("")


@pytest.fixture
def fake_loaders(monkeypatch):
    """Replace the antiSMASH/BiG-SLiCE-dependent loaders, recording their arguments."""
    calls = {}

    def fake_make_sparse_dict(antismash_folder, hmmdb_folder, num_cpus=0, glob_pattern=None, **kwargs):
        calls["make_sparse_dict"] = (antismash_folder, glob_pattern)
        return {f"{antismash_folder}/{name}.gbk": {"PF00001": 255} for name in ("a", "b")}

    def fake_parse_cb_outputs(target_dir, glob_pattern=None, num_cpus=0, mibig_mode=False):
        calls["parse_cb_outputs"] = (target_dir, glob_pattern)
        return {
            f"{target_dir}/{name}.txt": {"BGC0000001": 100, "BGC0000002": 50}
            for name in ("a", "b")
        }

    def fake_make_reference_polymer_dict(ref_dir, k=5, as6=False):
        calls["make_reference_polymer_dict"] = ref_dir
        return {"a": {"1_Ser"}, "b": {"1_Ser", "2_Ser"}}

    monkeypatch.setattr(custom_refs, "make_sparse_dict", fake_make_sparse_dict)
    monkeypatch.setattr(custom_refs, "parse_cb_outputs", fake_parse_cb_outputs)
    monkeypatch.setattr(custom_refs, "make_reference_polymer_dict", fake_make_reference_polymer_dict)
    return calls


# --- file name validation ---------------------------------------------------

def test_matching_names_pass(tmp_path):
    touch(tmp_path, "a.gbk", "a.txt", "a.json")
    assert check_reference_files(str(tmp_path), poly_search=True) == [str(tmp_path / "a.gbk")]


def test_antismash_default_names_fail_with_clear_message(tmp_path):
    touch(tmp_path, "a.region001.gbk", "a_c1.txt")
    with pytest.raises(CustomReferenceError) as excinfo:
        check_reference_files(str(tmp_path))
    message = str(excinfo.value)
    assert "all sharing the same <name>" in message
    assert "no .txt for a.region001" in message
    assert "no .gbk for a_c1.txt" in message


def test_default_names_fail_before_any_vectors_are_built(tmp_path, fake_loaders):
    touch(tmp_path, "a.region001.gbk", "a_c1.txt")
    with pytest.raises(CustomReferenceError):
        load_custom_references(str(tmp_path), "db")
    assert fake_loaders == {}


def test_json_not_required_without_poly_search(tmp_path):
    touch(tmp_path, "a.gbk", "a.txt")
    assert check_reference_files(str(tmp_path), poly_search=False) == [str(tmp_path / "a.gbk")]


def test_json_required_with_poly_search(tmp_path):
    touch(tmp_path, "a.gbk", "a.txt")
    with pytest.raises(CustomReferenceError, match=r"no \.json for a"):
        check_reference_files(str(tmp_path), poly_search=True)


def test_mismatched_json_name_fails(tmp_path):
    touch(tmp_path, "a.gbk", "a.txt", "a_record.json")
    with pytest.raises(CustomReferenceError, match=r"no \.gbk for a_record\.json"):
        check_reference_files(str(tmp_path), poly_search=True)


def test_directory_without_gbk_fails(tmp_path):
    touch(tmp_path, "a.txt")
    with pytest.raises(CustomReferenceError, match=r"no \.gbk files"):
        check_reference_files(str(tmp_path))


# --- vector validation --------------------------------------------------------

def test_reference_without_pfam_vector_fails():
    with pytest.raises(CustomReferenceError, match=r"no biosynthetic Pfam domains .* b\.gbk"):
        validate_reference_vectors(
            "refs", ["a", "b"],
            bs_vecs={"a": {"PF00001": 255}},
            cb_vecs={"a": {"BGC0000001": 1}, "b": {"BGC0000001": 1}},
        )


def test_unparsed_cb_file_fails_without_key_error():
    with pytest.raises(CustomReferenceError, match=r"no \.txt for b"):
        validate_reference_vectors(
            "refs", ["a", "b"],
            bs_vecs={"a": {"x": 1}, "b": {"x": 1}},
            cb_vecs={"a": {"BGC0000001": 1}},
        )


def test_missing_polymer_vector_fails_only_with_poly_search():
    kwargs = dict(
        bs_vecs={"a": {"x": 1}},
        cb_vecs={"a": {"BGC0000001": 1}},
        pol_vecs={},
    )
    validate_reference_vectors("refs", ["a"], poly_search=False, **kwargs)
    with pytest.raises(CustomReferenceError, match=r"no \.json for a"):
        validate_reference_vectors("refs", ["a"], poly_search=True, **kwargs)


def test_empty_cb_vector_warns_but_passes(capsys):
    validate_reference_vectors(
        "refs", ["a"], bs_vecs={"a": {"x": 1}}, cb_vecs={"a": {}},
    )
    assert "no KnownClusterBlast hits were read from a.txt" in capsys.readouterr().out


# --- directory handling -----------------------------------------------------

def test_loaders_receive_the_directory_path(tmp_path, fake_loaders):
    touch(tmp_path, "a.gbk", "a.txt", "b.gbk", "b.txt")
    ref_dir = str(tmp_path)

    labels, gbk_paths, bs_vecs, cb_vecs, pol_vecs = load_custom_references(ref_dir, "db")

    assert fake_loaders["make_sparse_dict"] == (ref_dir, "*.gbk")
    assert fake_loaders["parse_cb_outputs"] == (ref_dir, "*.txt")
    assert isinstance(fake_loaders["make_sparse_dict"][0], str)
    assert isinstance(fake_loaders["parse_cb_outputs"][0], str)
    # Polymer references are only built with --poly_search.
    assert "make_reference_polymer_dict" not in fake_loaders
    assert pol_vecs == {}

    assert labels == ["a", "b"]
    assert gbk_paths == [str(tmp_path / "a.gbk"), str(tmp_path / "b.gbk")]
    assert set(bs_vecs) == set(cb_vecs) == {"a", "b"}


def test_poly_search_builds_polymer_references(tmp_path, fake_loaders):
    touch(tmp_path, "a.gbk", "a.txt", "a.json", "b.gbk", "b.txt", "b.json")
    *_, pol_vecs = load_custom_references(str(tmp_path), "db", poly_search=True)
    assert fake_loaders["make_reference_polymer_dict"] == str(tmp_path)
    assert all(isinstance(value, set) for value in pol_vecs.values())


def test_mibig_exclude_removed_from_reference_cb_vectors(tmp_path, fake_loaders):
    touch(tmp_path, "a.gbk", "a.txt", "b.gbk", "b.txt")
    *_, cb_vecs, _ = load_custom_references(str(tmp_path), "db", mibig_exclude={"BGC0000001"})
    assert all(vec == {"BGC0000002": 50} for vec in cb_vecs.values())


def test_mibig_exclude_self_is_ignored(tmp_path, fake_loaders):
    touch(tmp_path, "a.gbk", "a.txt", "b.gbk", "b.txt")
    *_, cb_vecs, _ = load_custom_references(str(tmp_path), "db", mibig_exclude="self")
    assert all("BGC0000001" in vec for vec in cb_vecs.values())


def test_single_reference_warns(tmp_path, fake_loaders, monkeypatch, capsys):
    monkeypatch.setattr(
        custom_refs, "make_sparse_dict",
        lambda folder, db, num_cpus=0, glob_pattern=None: {f"{folder}/a.gbk": {"x": 1}},
    )
    monkeypatch.setattr(
        custom_refs, "parse_cb_outputs",
        lambda folder, glob_pattern=None, num_cpus=0: {f"{folder}/a.txt": {"BGC0000001": 1}},
    )
    touch(tmp_path, "a.gbk", "a.txt")
    labels, *_ = load_custom_references(str(tmp_path), "db")
    assert labels == ["a"]
    assert "only one custom reference BGC" in capsys.readouterr().out


# --- maximum k --------------------------------------------------------------

@pytest.mark.parametrize(
    "custom_refs_used, poly_search, max_k",
    [
        (False, False, 3),  # MIBiG: pfam-vec, blast-vec, BLAST-rank
        (False, True, 4),   # MIBiG + polymer
        (True, False, 2),   # custom: no BLAST-rank
        (True, True, 3),    # custom + polymer
    ],
)
def test_max_k_and_min_k_cap(custom_refs_used, poly_search, max_k):
    assert max_achievable_k(custom_refs_used, poly_search) == max_k
    # Default --min_k 3 is capped only where it cannot be reached.
    assert cap_min_k(3, custom_refs_used, poly_search) == (min(3, max_k), max_k)
    # --min_k 4 is always capped to the maximum.
    assert cap_min_k(4, custom_refs_used, poly_search) == (max_k, max_k)
    # Lower values are left alone.
    assert cap_min_k(1, custom_refs_used, poly_search) == (1, max_k)
