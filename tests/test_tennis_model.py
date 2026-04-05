# Tests for tennis_model: closed forms, normalization, symmetry.

from __future__ import annotations

import math

import pytest

import tennis_model as tm


def test_no_ad_win_prob_matches_closed_form_grid() -> None:
    for p in [0.0, 0.05, 0.25, 0.5, 0.62, 0.9, 1.0]:
        tm.clear_caches()
        closed = tm.no_ad_game_win_prob_closed_form(p)
        d = tm._noad_game_rec(0, 0, p)
        dp = float(sum(d[i] for i in (4, 5, 6, 7)))
        assert closed == pytest.approx(dp, rel=1e-12, abs=1e-12)
        assert closed == pytest.approx(dp, rel=0, abs=1e-14)


def test_no_ad_margin_pmf_matches_dp_and_sums_to_one() -> None:
    for p in [0.1, 0.37, 0.5, 0.73, 0.99]:
        tm.clear_caches()
        closed = tm.no_ad_margin_pmf_closed_form(p)
        dp = tm._noad_game_rec(0, 0, p)
        assert math.isclose(sum(closed), 1.0, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(sum(dp), 1.0, rel_tol=0, abs_tol=1e-12)
        for a, b in zip(closed, dp):
            assert a == pytest.approx(b, rel=1e-12, abs=1e-12)


def test_advantage_win_from_deuce_closed_form() -> None:
    for p in [0.01, 0.2, 0.5, 0.8, 0.99]:
        q = 1.0 - p
        d = tm.advantage_win_from_deuce_closed_form(p)
        tm.clear_caches()
        m_deuce, _ = tm.win_by_two_race_solve(4, 3, lambda a, b: p)
        from_deuce_dp = float(sum(m_deuce[(3, 3)][i] for i in (4, 5, 6, 7)))
        assert d == pytest.approx(p**2 / (p**2 + q**2), rel=1e-15)
        assert from_deuce_dp == pytest.approx(d, rel=1e-12)


def test_advantage_full_game_matches_polynomial_closed_form() -> None:
    for p in [0.02, 0.11, 0.35, 0.5, 0.66, 0.91, 0.998]:
        tm.clear_caches()
        poly = tm.advantage_game_win_prob_closed_form_polynomial(p)
        dp = tm.advantage_game_win_prob_iid(p)
        assert poly == pytest.approx(dp, rel=1e-11, abs=1e-11)


def test_advantage_game_symmetric_at_p_half() -> None:
    tm.clear_caches()
    p = 0.5
    assert tm.advantage_game_win_prob_iid(p) == pytest.approx(0.5, abs=1e-12)


def test_game_margin_distributions_sum_to_one() -> None:
    for p in [0.33, 0.71]:
        tm.clear_caches()
        da = tm._advantage_game_tables(p)[0][(0, 0)]
        dn = tm._noad_game_rec(0, 0, p)
        assert math.isclose(sum(da), 1.0, abs_tol=1e-12)
        assert math.isclose(sum(dn), 1.0, abs_tol=1e-12)


def test_set_distribution_sums_to_one() -> None:
    spec = tm.SetSpec(6, 6, 7, False)
    tm.clear_caches()
    d = tm.set_score_distribution(spec, True, 0.55, 0.45)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-10)


def test_fast4_set_sums_to_one() -> None:
    spec = tm.SetSpec(4, 3, 7, True)
    tm.clear_caches()
    d = tm.set_score_distribution(spec, False, 0.52, 0.48)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-10)


def test_match_distributions_sum_to_one() -> None:
    ps, pr, fa = 0.58, 0.41, True
    standard = tm.SetSpec(6, 6, 7, False)
    slam5_last = tm.SetSpec(6, 6, 10, False)
    tm.clear_caches()
    d1 = tm._match_bo3_three_sets(standard, fa, ps, pr)
    assert math.isclose(sum(d1.values()), 1.0, abs_tol=1e-9)
    tm.clear_caches()
    d2 = tm._match_bo3_mtb10(standard, fa, ps, pr)
    assert math.isclose(sum(d2.values()), 1.0, abs_tol=1e-9)
    tm.clear_caches()
    d3 = tm._slam_bo3_variable_tb((standard, standard, slam5_last), fa, ps, pr)
    assert math.isclose(sum(d3.values()), 1.0, abs_tol=1e-9)
    tm.clear_caches()
    d4 = tm._match_bo5_variable_last(standard, slam5_last, fa, ps, pr)
    assert math.isclose(sum(d4.values()), 1.0, abs_tol=1e-9)
    tm.clear_caches()
    d5 = tm._match_bo5_uniform(tm.SetSpec(4, 3, 7, True), fa, ps, pr)
    assert math.isclose(sum(d5.values()), 1.0, abs_tol=1e-9)


def test_first_server_does_not_change_match_level_distribution() -> None:
    """Alternating serve + ITF set-2 opener implies match win and set-count PMF are phase-invariant."""
    spec = tm.SetSpec(6, 6, 7, False)
    ps, pr = 0.71, 0.34
    tm.clear_caches()
    m_a = tm._match_bo3_three_sets(spec, True, ps, pr)
    tm.clear_caches()
    m_b = tm._match_bo3_three_sets(spec, False, ps, pr)
    assert m_a == pytest.approx(m_b, abs=1e-12)


def test_complementary_p_serve_p_return_yields_symmetric_match() -> None:
    """When p_serve + p_return = 1, alternating serve makes A/B mirror images → 50% set/match."""
    standard = tm.SetSpec(6, 6, 7, False)
    tm.clear_caches()
    s = tm.set_score_distribution(standard, True, 0.62, 0.38)
    pa_set = sum(p for (a, b), p in s.items() if a > b)
    assert pa_set == pytest.approx(0.5, abs=1e-9)
    tm.clear_caches()
    m = tm._match_bo3_three_sets(standard, True, 0.62, 0.38)
    pa_match = sum(p for (a, b), p in m.items() if a > b)
    assert pa_match == pytest.approx(0.5, abs=1e-9)
    assert m[(2, 0)] == pytest.approx(m[(0, 2)], abs=1e-9)
    assert m[(2, 1)] == pytest.approx(m[(1, 2)], abs=1e-9)


def test_match_row_win_rate_matches_distribution() -> None:
    ps, pr = 0.6, 0.4
    standard = tm.SetSpec(6, 6, 7, False)
    tm.clear_caches()
    dist = tm._match_bo3_three_sets(standard, True, ps, pr)
    row = tm._dist_to_match_row(dist, False)
    pa = sum(p for (sa, sb), p in dist.items() if sa > sb)
    assert row["Player A win %"] == pytest.approx(pa, abs=1e-10)


def test_tiebreak_tb7_extreme_servers() -> None:
    tm.clear_caches()
    d, _ = tm._tiebreak_solve(7, 6, True, 1.0, 0.0)
    assert math.isclose(sum(d), 1.0, abs_tol=1e-12)
    assert tm._play_tiebreak_win_prob_a(7, True, 1.0, 0.0) >= 0.0


def test_tiebreak_serve_rotation_first_server_points() -> None:
    """ITF: starter serves pt 1, then opponent 2–3, starter 4–5, … → 1,4,5,8,9,12,13,…"""
    first_pts = {1, 4, 5, 8, 9, 12, 13, 16, 17}
    second_pts = {2, 3, 6, 7, 10, 11, 14, 15, 18}
    for pt in range(1, 19):
        is_a = tm.tiebreak_server_is_a(pt, True)
        assert is_a == (pt in first_pts)
        assert pt in first_pts or pt in second_pts
    for pt in range(1, 19):
        assert tm.tiebreak_server_is_a(pt, False) is (not tm.tiebreak_server_is_a(pt, True))


def test_game_primitives_table_shape() -> None:
    tm.clear_caches()
    df = tm.game_primitives_table(0.6, 0.35)
    assert len(df) == 5
    assert list(df.columns[0:4]) == [
        tm.GAME_TABLE_ROW_LABEL_COL,
        "Player A win %",
        "Player B win %",
        tm.GAME_DEUCE_COL_TOP,
    ]
    idx = tm.GAME_PRIMITIVES_MID_HEADER_ROW_INDEX
    assert df.iloc[idx].tolist() == list(tm.GAME_PRIMITIVES_MID_HEADER_LABELS)


def test_match_formats_table_row_labels() -> None:
    tm.clear_caches()
    df = tm.match_formats_table(0.55, 0.44, True)
    assert len(df) == 5
    assert "Grand Slam Men's Singles" in set(df[tm.MATCH_TABLE_ROW_LABEL_COL])
