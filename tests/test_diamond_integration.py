"""
End-to-end check of the diamond blast source on a tiny made-up database.
Needs the 'diamond' executable; skipped otherwise.
"""

import random
import shutil

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from discern import homology
from discern.cb_sources import diamond_cb_vectors
from discern.custom_refs import CustomReferenceError
from discern.funcs_2 import find_hits_intersection
from discern.mibig_db import build_set_from_proteins

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("diamond") is None, reason="diamond not installed"),
]

AMINO = "ACDEFGHIKLMNPQRSTVWY"


def protein(rng, n=250):
    return "M" + "".join(rng.choice(AMINO) for _ in range(n - 1))


def mutate(rng, seq, fraction):
    seq = list(seq)
    for i in rng.sample(range(1, len(seq)), int(fraction * len(seq))):
        seq[i] = rng.choice(AMINO.replace(seq[i], ""))
    return "".join(seq)


def write_region_gbk(path, proteins):
    record = SeqRecord(Seq("A" * 1000 * len(proteins)), id="rec", name="rec",
                       annotations={"molecule_type": "DNA"})
    for i, seq in enumerate(proteins):
        record.features.append(SeqFeature(FeatureLocation(i * 1000, i * 1000 + 900, strand=1),
                                          type="CDS", qualifiers={"translation": [seq]}))
    SeqIO.write(record, str(path), "genbank")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = random.Random(1)
    p1, p2, p3, p4, p5, p6 = (protein(rng) for _ in range(6))
    mibig = {
        "BGC0000001": [p1, p2, p3],
        "BGC0000002": [mutate(rng, p1, 0.2), mutate(rng, p2, 0.2), p4],  # ~80 % identical copies
        "BGC0000003": [p1, p5],     # shares only one gene with the query
        "BGC0000004": [p6],         # single-gene BGC
    }
    root = tmp_path_factory.mktemp("diamond")
    mibig_set = build_set_from_proteins("4.0", mibig, {k: "4.0" for k in mibig}, root / "db" / "4.0", threads=2)

    antismash_dir = root / "antismash"
    (antismash_dir / "genome").mkdir(parents=True)
    query = antismash_dir / "genome" / "contig.region001.gbk"
    write_region_gbk(query, [p1, p2, p6, protein(rng)])

    custom = root / "custom"
    custom.mkdir()
    write_region_gbk(custom / "my_ref.gbk", [p1, p2, p4])
    return mibig_set, str(antismash_dir), str(query), str(custom / "my_ref.gbk"), root


def self_score(mibig_set, bgc):
    return mibig_set.background_raw()[bgc][bgc][0]


def test_background_and_query_scores(setup):
    mibig_set, antismash_dir, query, _, root = setup
    raw = mibig_set.background_raw()
    assert raw["BGC0000001"]["BGC0000001"][1:] == [3, 3]   # self hit on all three genes

    queries, refs = diamond_cb_vectors(antismash_dir, mibig_set, str(root / "out1"), threads=2)
    vec = queries[query]
    assert vec["BGC0000001"] > vec["BGC0000002"] > 0   # identical genes beat ~80 % copies
    assert "BGC0000003" not in vec                     # one query gene only: fails the two-gene rule
    assert vec["BGC0000004"] == self_score(mibig_set, "BGC0000004")   # single-gene BGC is exempt
    assert list(vec) == sorted(vec, key=lambda s: -vec[s])            # ordered by score
    assert set(refs) == {"BGC0000001", "BGC0000002", "BGC0000003", "BGC0000004"}


def test_query_score_equals_sum_of_self_hits(setup):
    mibig_set, antismash_dir, query, _, root = setup
    queries, _ = diamond_cb_vectors(antismash_dir, mibig_set, str(root / "out2"), threads=2)
    # p1 and p2 are identical in the query and in BGC0000001, so each query gene's
    # best hit is the self hit of that protein
    fasta = root / "p12.fa"
    p1_p2 = mibig_set.proteins_by_bgc()["BGC0000001"][:2]
    homology.write_bgc_fasta([("X", p1_p2)], str(fasta))
    db = homology.make_diamond_db(str(fasta), str(root / "p12"))
    expected = homology.search(str(fasta), db, str(root / "p12_hits"))["X"]["X"][0]
    assert queries[query]["BGC0000001"] == expected


def test_custom_references_become_subjects(setup):
    mibig_set, antismash_dir, query, custom_gbk, root = setup
    queries, refs = diamond_cb_vectors(antismash_dir, mibig_set, str(root / "out3"), threads=2,
                                       custom_refs={"my_ref": custom_gbk})
    assert "my_ref" in queries[query]                 # custom dimension in the query vector
    assert refs["my_ref"]["my_ref"] > 0               # custom row with its self hit
    assert "BGC0000001" in refs["my_ref"]             # custom row against MIBiG
    assert "my_ref" in refs["BGC0000001"]             # MIBiG row against the custom reference
    # BLAST-rank with a mixed reference set, self dropped by ID
    hits = find_hits_intersection(refs, queries, ["BGC0000001", "my_ref"], drop_self_by_id=True)
    assert hits == [query]


def test_exclusion_removes_rows_and_dimensions(setup):
    mibig_set, antismash_dir, query, _, root = setup
    queries, refs = diamond_cb_vectors(antismash_dir, mibig_set, str(root / "out4"), threads=2,
                                       exclude={"BGC0000002"})
    assert "BGC0000002" not in queries[query]
    assert "BGC0000002" not in refs
    assert all("BGC0000002" not in vec for vec in refs.values())


def test_reuse_skips_query_search(setup, capsys):
    mibig_set, antismash_dir, query, _, root = setup
    first, _ = diamond_cb_vectors(antismash_dir, mibig_set, str(root / "out5"), threads=2)
    capsys.readouterr()
    again, _ = diamond_cb_vectors(antismash_dir, mibig_set, str(root / "out6"), threads=2,
                                  reuse_dir=str(root / "out5"))
    assert "loading previously computed DIAMOND hits" in capsys.readouterr().out
    assert again == first


def test_custom_name_clash_rejected(setup):
    mibig_set, antismash_dir, _, custom_gbk, root = setup
    with pytest.raises(CustomReferenceError, match="clash with MIBiG IDs"):
        diamond_cb_vectors(antismash_dir, mibig_set, str(root / "out7"), threads=2,
                           custom_refs={"BGC0000001": custom_gbk})
