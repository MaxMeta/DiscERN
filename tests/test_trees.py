from discern.vector_tools import get_vecs_for_trees, make_trees


def test_make_trees_skips_with_fewer_than_two_items(tmp_path, capsys):
    make_trees(str(tmp_path), [], ["only_ref"], {}, {}, {}, {}, {}, {}, use_pol=True)
    assert "skipping tree building" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_get_vecs_for_trees_uses_capped_min_k():
    hits = {1: {"k1"}, 2: {"k2"}, 3: set(), 4: set()}
    # Custom references without --poly_search: k=2 is the most a hit can reach.
    assert sorted(get_vecs_for_trees(hits, min_k=2)) == ["k2"]
    assert get_vecs_for_trees(hits, min_k=3) == []
