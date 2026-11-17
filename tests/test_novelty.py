import numpy as np
import pytest

from marlio.agents.neat.novelty import NoveltyArchive, _ranks


def test_ranks_share_ties():
    assert list(_ranks(np.array([3.0, 1.0, 3.0, 2.0]))) == [3.5, 1, 3.5, 2]


def paths():
    # Three runs that end at the same spot and one that goes somewhere else.
    same = [0.0, 0.0, 1.0, 0.0]
    return [same, same, same, [5.0, 3.0, 9.0, 3.0]]


def test_the_odd_run_is_the_most_novel():
    archive = NoveltyArchive(k=2)
    out = archive.select_scores(paths(), [1, 2, 3, 0], "nslc")
    assert out["novelty"].argmax() == 3
    assert len(archive) > 0


def test_fitness_selection_keeps_fitness():
    out = NoveltyArchive(k=2).select_scores(paths(), [1, 2, 3, 0], "fitness")
    assert list(out["score"]) == [1, 2, 3, 0]


def test_novelty_lifts_a_weak_but_new_run():
    fit = [1, 2, 3, 0]
    plain = _ranks(np.array(fit, dtype=float))
    out = NoveltyArchive(k=2).select_scores(paths(), fit, "nslc", novelty_weight=0.5)
    assert out["score"][3] > plain[3]


def test_dns_best_scores_top_and_copies_of_better_score_low():
    out = NoveltyArchive(k=2).select_scores(paths(), [1, 2, 3, 0], "dns")
    dominated = out["dominated_novelty"]
    assert dominated[2] == dominated.max()  # the best run
    assert dominated[1] == 0  # identical to a better run
    assert all(out["score"] > 0)


def test_archive_save_and_load(tmp_path):
    archive = NoveltyArchive(k=3)
    archive.select_scores(paths(), [1, 2, 3, 0], "dns")
    archive.save(tmp_path / "a.json")
    again = NoveltyArchive.load(tmp_path / "a.json")
    assert again.behaviors == archive.behaviors and again.k == 3


def test_unknown_selection():
    with pytest.raises(ValueError):
        NoveltyArchive().select_scores(paths(), [1, 2, 3, 0], "map-elites")
