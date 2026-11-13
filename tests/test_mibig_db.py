"""Merging MIBiG releases into reference sets (discern.mibig_db), offline."""

import pytest

from discern.mibig_db import (
    MIBIG_SETS,
    MibigSetError,
    RELEASE_FASTAS,
    load_set,
    md5sum,
    merge_releases,
    read_release_fasta,
    unversioned,
)


def test_unversioned():
    assert unversioned("BGC0000001.5") == "BGC0000001"
    assert unversioned("BGC0000001") == "BGC0000001"


def test_read_release_fasta_strips_versions_and_keeps_order(tmp_path):
    fasta = tmp_path / "p.fasta"
    fasta.write_text(
        ">BGC0000001.5|c1|1-9|+|A|x|A\nMKV\nLL\n"
        ">BGC0000001.5|c1|9-20|+|B|y|B\nMAA\n"
        ">BGC0000002.1|c1|1-9|+|C|z|C\n\n"          # empty sequence is skipped
        ">BGC0000003.1|c1|1-9|+|D|z|D\nMQQ\n"
    )
    assert read_release_fasta(fasta) == {"BGC0000001": ["MKVLL", "MAA"], "BGC0000003": ["MQQ"]}


def test_merge_takes_each_bgc_from_highest_priority_release():
    releases = [
        ("4.0", {"BGC0000002": ["NEW"], "BGC0000003": ["N3"]}),
        ("3.1", {"BGC0000001": ["OLD31"], "BGC0000002": ["OLD"]}),
        ("1.4", {"BGC0000001": ["OLD14"], "BGC0000009": ["X"]}),
    ]
    merged, origin = merge_releases(releases)
    assert list(merged) == ["BGC0000001", "BGC0000002", "BGC0000003", "BGC0000009"]
    assert merged["BGC0000002"] == ["NEW"]
    assert merged["BGC0000001"] == ["OLD31"]
    assert origin == {"BGC0000001": "3.1", "BGC0000002": "4.0", "BGC0000003": "4.0", "BGC0000009": "1.4"}


def test_sets_are_defined_from_pinned_releases():
    assert MIBIG_SETS["4.0"] == ("4.0",)
    assert MIBIG_SETS["4.0_legacy"][0] == "4.0"
    assert set(MIBIG_SETS["4.0_legacy"]) <= set(RELEASE_FASTAS)
    assert all(len(md5) == 32 for _, md5 in RELEASE_FASTAS.values())


def test_md5sum(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert md5sum(path) == "900150983cd24fb0d6963f7d28e17f72"


def test_missing_set_explains_how_to_build_it(tmp_path):
    with pytest.raises(MibigSetError, match="discern-setup-mibig --set 4.0_legacy"):
        load_set("4.0_legacy", root=tmp_path)


def test_unknown_set_rejected(tmp_path):
    with pytest.raises(MibigSetError, match="unknown MIBiG set"):
        load_set("3.1", root=tmp_path)
