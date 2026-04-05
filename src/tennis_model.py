# Exact tennis probability primitives and match presets (Markov / memoized DP).

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import pandas as pd

sys.setrecursionlimit(20000)

# 0-based index of the repeated column-title row in `game_primitives_table` (between games and tiebreaks).
GAME_PRIMITIVES_MID_HEADER_ROW_INDEX = 2

# First column of ``game_primitives_table`` / ``match_formats_table``.
GAME_TABLE_ROW_LABEL_COL = "Single game"
MATCH_TABLE_ROW_LABEL_COL = "Match Format"

# Game / tiebreak table: header label (tiebreak rows still show P(extra pts) in that column).
GAME_DEUCE_COL_HEADER = "Prob(deuce)"
# Mid-table gray header row: clarifies tiebreak rows use the “extra pts” interpretation.
GAME_DEUCE_COL_MID = "Prob(extra pts)"

GAME_PRIMITIVES_MID_HEADER_LABELS: List[str] = [
    "Tiebreak",
    "",
    "",
    GAME_DEUCE_COL_MID,
]

MARGIN_COLS = [
    "Lose @ 0-40",
    "Lose @ 15-40",
    "Lose @ 30-40",
    "Lose @ Deuce",
    "Win @ Deuce",
    "Win @ 40-30",
    "Win @ 40-15",
    "Win @ 40-0",
]

# En dash for set scores in match-win equation cells, e.g. "(3–0)".
_MATCH_SCORE_SEP = "\u2013"


def no_ad_game_win_prob_closed_form(p: float) -> float:
    """
    P(A wins a no-ad game) when A wins each point i.i.d. with prob p (A always serving).
    Direct counting: 4-0, 4-1, 4-2, or 4-3 via deciding point at 3-3.
    = p^4 * (1 + 4(1-p) + 10(1-p)^2 + 20(1-p)^3).
    """
    q = 1.0 - p
    return (p**4) * (1.0 + 4 * q + 10 * q**2 + 20 * q**3)


def advantage_win_from_deuce_closed_form(p: float) -> float:
    """P(A wins from deuce) with i.i.d. point win prob p for A on every point."""
    return (p**2) / (p**2 + (1.0 - p) ** 2)


def _bucket_for_margin(m: int) -> int:
    if m <= -4:
        return 0
    if m == -3:
        return 1
    if m == -2:
        return 2
    if m == -1:
        return 3
    if m == 1:
        return 4
    if m == 2:
        return 5
    if m == 3:
        return 6
    return 7


def _merge_dist(a: Tuple[float, ...], b: Tuple[float, ...], wa: float, wb: float) -> Tuple[float, ...]:
    return tuple(wa * x + wb * y for x, y in zip(a, b))


def _zero_dist() -> Tuple[float, ...]:
    return tuple(0.0 for _ in MARGIN_COLS)


def _terminal_margin(m: int) -> Tuple[float, ...]:
    d = list(_zero_dist())
    d[_bucket_for_margin(m)] = 1.0
    return tuple(d)


def tiebreak_server_is_a(point_1_indexed: int, first_server_is_a: bool) -> bool:
    if point_1_indexed == 1:
        return first_server_is_a
    block = (point_1_indexed - 2) // 2
    return first_server_is_a if (block % 2 == 1) else (not first_server_is_a)


def p_a_wins_point_on_tiebreak(
    a_pts: int,
    b_pts: int,
    first_tb_server_is_a: bool,
    p_serve: float,
    p_return: float,
) -> float:
    pt = a_pts + b_pts + 1
    return p_serve if tiebreak_server_is_a(pt, first_tb_server_is_a) else p_return


def _race_terminal(a: int, b: int, target: int) -> bool:
    """First to `target` points, win by 2 (same rule for a game with target=4 and tiebreaks with target 7 or 10)."""
    if a >= target and a - b >= 2:
        return True
    if b >= target and b - a >= 2:
        return True
    return False


def win_by_two_race_solve(
    target: int,
    tie_visit_level: int,
    p_a_wins_next: Callable[[int, int], float],
) -> Tuple[Dict[Tuple[int, int], Tuple[float, ...]], Dict[Tuple[int, int], float]]:
    """
    Distributions for a point race: first to `target` with win-by-2.

    A regular tennis game (advantage scoring) is the target=4 case with constant P(A wins point).
    A tiebreak to 7 or 10 is the same rule with target=7 or 10 and a state-dependent point win
    probability (alternating serve).

    Returns (margin_pmf_by_state, tie_visit_prob_by_state); use [...][(0, 0)] from the start.
    """
    dim = max(220, target * 2 + 30)
    margin: Dict[Tuple[int, int], Tuple[float, ...]] = {}
    visit: Dict[Tuple[int, int], float] = {}

    def mterm(aa: int, bb: int) -> Tuple[float, ...]:
        return _terminal_margin(aa - bb)

    def child_margin(aa: int, bb: int) -> Tuple[float, ...]:
        if _race_terminal(aa, bb, target):
            return mterm(aa, bb)
        if aa > dim or bb > dim:
            if aa - bb >= 2:
                return mterm(aa, bb)
            if bb - aa >= 2:
                return mterm(aa, bb)
            return _merge_dist(_terminal_margin(2), _terminal_margin(-2), 0.5, 0.5)

        return margin[(aa, bb)]

    def child_visit(aa: int, bb: int) -> float:
        if _race_terminal(aa, bb, target):
            return 0.0
        if aa > dim or bb > dim:
            return 0.0
        return visit[(aa, bb)]

    for a in range(dim + 1):
        for b in range(dim + 1):
            if _race_terminal(a, b, target):
                margin[(a, b)] = mterm(a, b)
                visit[(a, b)] = 0.0

    for s in range(2 * dim, -1, -1):
        for a in range(max(0, s - dim), min(dim, s) + 1):
            b = s - a
            if (a, b) in margin:
                continue
            pwin = p_a_wins_next(a, b)
            d1 = child_margin(a + 1, b)
            d2 = child_margin(a, b + 1)
            margin[(a, b)] = _merge_dist(d1, d2, pwin, 1.0 - pwin)
            v1 = child_visit(a + 1, b)
            v2 = child_visit(a, b + 1)
            if a == tie_visit_level and b == tie_visit_level:
                visit[(a, b)] = 1.0
            else:
                visit[(a, b)] = pwin * v1 + (1.0 - pwin) * v2

    return margin, visit


@lru_cache(maxsize=None)
def _tiebreak_solve(
    target: int,
    tie_level: int,
    first_a: bool,
    p_serve: float,
    p_return: float,
) -> Tuple[Tuple[float, ...], float]:
    def p_next(a: int, b: int) -> float:
        return p_a_wins_point_on_tiebreak(a, b, first_a, p_serve, p_return)

    m, v = win_by_two_race_solve(target, tie_level, p_next)
    return m[(0, 0)], v[(0, 0)]


@lru_cache(maxsize=None)
def _advantage_game_tables(p: float) -> Tuple[Dict[Tuple[int, int], Tuple[float, ...]], Dict[Tuple[int, int], float]]:
    """Advantage game = win-by-two race to 4 points; A serves every point (constant p)."""
    return win_by_two_race_solve(4, 3, lambda a, b: p)


@lru_cache(maxsize=None)
def _p_adv_first_absorb_at(a: int, b: int, p: float, ta: int, tb: int) -> float:
    """
    P(first absorption is exactly (ta,tb) with **A** winning | start (a,b)), race to 4 win-by-2.
    ``cap`` truncates extreme deuce depth (negligible mass lost for (4,2) targets from (0,0)).
    """
    cap = 55
    if a > cap or b > cap:
        return 0.0
    if a >= 4 and a - b >= 2:
        return 1.0 if (a, b) == (ta, tb) else 0.0
    if b >= 4 and b - a >= 2:
        return 0.0
    return p * _p_adv_first_absorb_at(a + 1, b, p, ta, tb) + (1.0 - p) * _p_adv_first_absorb_at(
        a, b + 1, p, ta, tb
    )


@lru_cache(maxsize=None)
def _p_adv_first_absorb_b_win_at(a: int, b: int, p: float, ta: int, tb: int) -> float:
    """Same as above when **B** wins at the absorbing score (ta, tb)."""
    cap = 55
    if a > cap or b > cap:
        return 0.0
    if b >= 4 and b - a >= 2:
        return 1.0 if (a, b) == (ta, tb) else 0.0
    if a >= 4 and a - b >= 2:
        return 0.0
    return p * _p_adv_first_absorb_b_win_at(a + 1, b, p, ta, tb) + (1.0 - p) * _p_adv_first_absorb_b_win_at(
        a, b + 1, p, ta, tb
    )


def _advantage_margin2_split_a(p: float) -> Tuple[float, float]:
    """
    For A wins with point margin +2: (4,2) never visits 3–3 vs all other +2 terminals (deuce played).
    """
    m, _ = _advantage_game_tables(p)
    d5 = float(m[(0, 0)][5])
    p_hold = float(_p_adv_first_absorb_at(0, 0, p, 4, 2))
    p_ad = max(0.0, d5 - p_hold)
    return p_hold, p_ad


def _advantage_margin2_split_b(p: float) -> Tuple[float, float]:
    """B wins +2 margin: (2,4) before deuce vs (3,5),(4,6),…"""
    m, _ = _advantage_game_tables(p)
    d2 = float(m[(0, 0)][2])
    p_hold = float(_p_adv_first_absorb_b_win_at(0, 0, p, 2, 4))
    p_ad = max(0.0, d2 - p_hold)
    return p_hold, p_ad


@lru_cache(maxsize=None)
def _noad_game_rec(a: int, b: int, p: float) -> Tuple[float, ...]:
    if a >= 4 and a > b:
        return _terminal_margin(a - b)
    if b >= 4 and b > a:
        return _terminal_margin(a - b)
    if a == 3 and b == 3:
        return _merge_dist(_terminal_margin(1), _terminal_margin(-1), p, 1.0 - p)
    d1 = _noad_game_rec(a + 1, b, p)
    d2 = _noad_game_rec(a, b + 1, p)
    return _merge_dist(d1, d2, p, 1.0 - p)


@lru_cache(maxsize=None)
def _noad_deuce_visit_rec(a: int, b: int, p: float) -> float:
    if a >= 4 and a > b:
        return 0.0
    if b >= 4 and b > a:
        return 0.0
    if a == 3 and b == 3:
        return 1.0
    return p * _noad_deuce_visit_rec(a + 1, b, p) + (1.0 - p) * _noad_deuce_visit_rec(a, b + 1, p)


def _game_win_prob_ad(no_ad: bool, a_serves: bool, p_serve: float, p_return: float) -> float:
    p = p_serve if a_serves else p_return
    d = (
        _noad_game_rec(0, 0, p)
        if no_ad
        else _advantage_game_tables(p)[0][(0, 0)]
    )
    return float(sum(d[i] for i in (4, 5, 6, 7)))


def _play_tiebreak_win_prob_a(
    target: int,
    first_point_server_is_a: bool,
    p_serve: float,
    p_return: float,
) -> float:
    dist, _ = _tiebreak_solve(target, target - 1, first_point_server_is_a, p_serve, p_return)
    return float(sum(dist[i] for i in (4, 5, 6, 7)))


# (margin_dist index, display tag). A wins: strongest finish first; B wins: B’s POV tags.
_GAME_EQ_A_ORDER: Tuple[Tuple[int, str], ...] = (
    (7, "(@40-0)"),
    (6, "(@40-15)"),
    (5, "(@40-30)"),
    (4, "(@Deuce)"),
)
_GAME_EQ_B_ORDER: Tuple[Tuple[int, str], ...] = (
    (0, "(@40-0)"),
    (1, "(@40-15)"),
    (2, "(@40-30)"),
    (3, "(@Deuce)"),
)
# Tiebreak rows: same margin buckets as games, but labels are point-win margins (A or B).
_TB_EQ_A_ORDER: Tuple[Tuple[int, str], ...] = (
    (4, "(win by 1)"),
    (5, "(win by 2)"),
    (6, "(win by 3)"),
    (7, "(win by 4+)"),
)
_TB_EQ_B_ORDER: Tuple[Tuple[int, str], ...] = (
    (3, "(win by 1)"),
    (2, "(win by 2)"),
    (1, "(win by 3)"),
    (0, "(win by 4+)"),
)


def _game_win_equation(
    dist: Tuple[float, ...],
    *,
    for_a: bool,
    no_ad: bool,
    tiebreak: bool = False,
    adv_margin2_split_a: Tuple[float, float] | None = None,
    adv_margin2_split_b: Tuple[float, float] | None = None,
) -> str:
    """
    Games: ``32.9% = 3.1% (@40-0) + …`` from ``MARGIN_COLS`` path buckets.
    For **advantage** rows only, ``adv_margin2_split_*`` replaces the single +2 bucket with
    ``(@40-30)`` (no deuce) and ``(@Ad-in)`` (won after 3–3, still +2 points).
    Tiebreaks: point-margin labels (win by 1, …).
    """
    if tiebreak:
        order = _TB_EQ_A_ORDER if for_a else _TB_EQ_B_ORDER
        p_total = float(sum(dist[i] for i, _ in order))
        parts: list[str] = []
        for idx, tag in order:
            p = dist[idx]
            if p <= 0.0:
                continue
            parts.append(f"{100.0 * p:.1f}% {tag}")
        if not parts:
            return f"{100.0 * p_total:.1f}% ="
        return f"{100.0 * p_total:.1f}% = " + " + ".join(parts)

    if for_a:
        order = _GAME_EQ_A_ORDER
        p_total = float(sum(dist[i] for i, _ in order))
        parts = []
        use_split = adv_margin2_split_a is not None and not no_ad
        for idx, tag in order:
            if use_split and idx == 5:
                assert adv_margin2_split_a is not None
                ph, pa = adv_margin2_split_a
                if ph > 0.0:
                    parts.append(f"{100.0 * ph:.1f}% (@40-30)")
                if pa > 0.0:
                    parts.append(f"{100.0 * pa:.1f}% (@Ad-in)")
                continue
            p = dist[idx]
            if p <= 0.0:
                continue
            parts.append(f"{100.0 * p:.1f}% {tag}")
        if not parts:
            return f"{100.0 * p_total:.1f}% ="
        return f"{100.0 * p_total:.1f}% = " + " + ".join(parts)

    order = _GAME_EQ_B_ORDER
    p_total = float(sum(dist[i] for i, _ in order))
    parts = []
    use_split = adv_margin2_split_b is not None and not no_ad
    for idx, tag in order:
        if use_split and idx == 2:
            assert adv_margin2_split_b is not None
            ph, pa = adv_margin2_split_b
            if ph > 0.0:
                parts.append(f"{100.0 * ph:.1f}% (@40-30)")
            if pa > 0.0:
                parts.append(f"{100.0 * pa:.1f}% (@Ad-in)")
            continue
        p = dist[idx]
        if p <= 0.0:
            continue
        parts.append(f"{100.0 * p:.1f}% {tag}")
    if not parts:
        return f"{100.0 * p_total:.1f}% ="
    return f"{100.0 * p_total:.1f}% = " + " + ".join(parts)


def game_primitives_table(p_serve: float, p_return: float) -> pd.DataFrame:
    """
    Table 1: A serves deuce/no-ad games; A serves point 1 of each tiebreak.

    Columns: row label, **Player A wins** / **Player B wins** (equation cells), then
    ``GAME_DEUCE_COL_HEADER`` (P(ever deuce) for games; tiebreak rows use P(extra pts) in the same column).

    Inserts ``GAME_PRIMITIVES_MID_HEADER_ROW_INDEX`` (gray row) before tiebreak rows.
    """
    cols = [
        GAME_TABLE_ROW_LABEL_COL,
        "Player A wins",
        "Player B wins",
        GAME_DEUCE_COL_HEADER,
    ]
    data: list[list[object]] = []
    m_ad, v_ad = _advantage_game_tables(p_serve)
    dist_ad = m_ad[(0, 0)]
    sa = _advantage_margin2_split_a(p_serve)
    sb = _advantage_margin2_split_b(p_serve)
    data.append(
        [
            "A serves (with deuces)",
            _game_win_equation(
                dist_ad,
                for_a=True,
                no_ad=False,
                adv_margin2_split_a=sa,
            ),
            _game_win_equation(
                dist_ad,
                for_a=False,
                no_ad=False,
                adv_margin2_split_b=sb,
            ),
            v_ad[(0, 0)],
        ]
    )
    dist_na = _noad_game_rec(0, 0, p_serve)
    data.append(
        [
            "A serves (no-ad scoring)",
            _game_win_equation(dist_na, for_a=True, no_ad=True),
            _game_win_equation(dist_na, for_a=False, no_ad=True),
            _noad_deuce_visit_rec(0, 0, p_serve),
        ]
    )
    assert len(GAME_PRIMITIVES_MID_HEADER_LABELS) == len(cols)
    data.append(list(GAME_PRIMITIVES_MID_HEADER_LABELS))
    for target, label in ((7, "Tiebreak to 7"), (10, "Tiebreak to 10")):
        tie_level = target - 1
        dist, dt = _tiebreak_solve(target, tie_level, True, p_serve, p_return)
        data.append(
            [
                label,
                _game_win_equation(dist, for_a=True, no_ad=False, tiebreak=True),
                _game_win_equation(dist, for_a=False, no_ad=False, tiebreak=True),
                dt,
            ]
        )
    return pd.DataFrame(data, columns=cols)


@dataclass(frozen=True)
class SetSpec:
    games_to: int
    tb_at: int
    tb_target: int
    no_ad: bool = False


def _set_terminal(ga: int, gb: int, spec: SetSpec) -> bool:
    if ga >= spec.games_to and ga - gb >= 2:
        return True
    if gb >= spec.games_to and gb - ga >= 2:
        return True
    return False


def set_score_distribution(
    spec: SetSpec,
    first_game_server_is_a: bool,
    p_serve: float,
    p_return: float,
) -> Dict[Tuple[int, int, bool], float]:
    """
    Returns {(games_a, games_b, next_set_opener_is_a): probability}.

    next_set_opener_is_a follows the ATP/WTA continuation rule: whoever would have
    served the next game (had the set continued) serves game 1 of the next set.
    Equivalently, the opener flips iff the set contained an odd number of games.
    A tiebreak counts as one game, so a 7-6 set (13 games, odd) flips the opener.
    """
    states: Dict[Tuple[int, int, bool], float] = {(0, 0, first_game_server_is_a): 1.0}
    out: Dict[Tuple[int, int, bool], float] = defaultdict(float)

    while states:
        nxt: Dict[Tuple[int, int, bool], float] = defaultdict(float)
        for (ga, gb, srv_a), mass in list(states.items()):
            if mass == 0.0:
                continue
            if ga == spec.tb_at and gb == spec.tb_at:
                # Tiebreak is one game; srv_a serves it, so not srv_a opens next set.
                pw = _play_tiebreak_win_prob_a(spec.tb_target, srv_a, p_serve, p_return)
                next_opener = not srv_a
                out[(ga + 1, gb, next_opener)] += mass * pw
                out[(ga, gb + 1, next_opener)] += mass * (1.0 - pw)
                continue
            if _set_terminal(ga, gb, spec):
                # srv_a would serve the next game → srv_a opens the next set.
                out[(ga, gb, srv_a)] += mass
                continue
            pa = _game_win_prob_ad(spec.no_ad, srv_a, p_serve, p_return)
            pb = 1.0 - pa
            nxt[(ga + 1, gb, not srv_a)] += mass * pa
            nxt[(ga, gb + 1, not srv_a)] += mass * pb
        states = nxt
    return dict(out)


def _match_bo3_three_sets(
    set_spec: SetSpec,
    first_set_server_a: bool,
    p_serve: float,
    p_return: float,
) -> Dict[Tuple[int, int], float]:
    d_total: Dict[Tuple[int, int], float] = defaultdict(float)
    s1 = set_score_distribution(set_spec, first_set_server_a, p_serve, p_return)
    for (g1a, g1b, s2_opener), p1 in s1.items():
        a_won_s1 = g1a > g1b
        s2 = set_score_distribution(set_spec, s2_opener, p_serve, p_return)
        for (g2a, g2b, s3_opener), p2 in s2.items():
            p12 = p1 * p2
            sa = int(a_won_s1) + int(g2a > g2b)
            sb = int(not a_won_s1) + int(g2b > g2a)
            if sa == 2 or sb == 2:
                d_total[(sa, sb)] += p12
                continue
            s3 = set_score_distribution(set_spec, s3_opener, p_serve, p_return)
            for (g3a, g3b, _), p3 in s3.items():
                sa3 = sa + int(g3a > g3b)
                sb3 = sb + int(g3b > g3a)
                d_total[(sa3, sb3)] += p12 * p3
    return dict(d_total)


def _match_bo3_mtb10(
    set_spec: SetSpec,
    first_set_server_a: bool,
    p_serve: float,
    p_return: float,
) -> Dict[Tuple[int, int], float]:
    d_total: Dict[Tuple[int, int], float] = defaultdict(float)
    s1 = set_score_distribution(set_spec, first_set_server_a, p_serve, p_return)
    for (g1a, g1b, s2_opener), p1 in s1.items():
        a_won_s1 = g1a > g1b
        s2 = set_score_distribution(set_spec, s2_opener, p_serve, p_return)
        for (g2a, g2b, mtb_opener), p2 in s2.items():
            p12 = p1 * p2
            sa = int(a_won_s1) + int(g2a > g2b)
            sb = int(not a_won_s1) + int(g2b > g2a)
            if sa == 2:
                d_total[(2, 0)] += p12
                continue
            if sb == 2:
                d_total[(0, 2)] += p12
                continue
            pw = _play_tiebreak_win_prob_a(10, mtb_opener, p_serve, p_return)
            d_total[(2, 1)] += p12 * pw
            d_total[(1, 2)] += p12 * (1.0 - pw)
    return dict(d_total)


def _slam_bo3_variable_tb(
    set_specs: Tuple[SetSpec, SetSpec, SetSpec],
    first_set_server_a: bool,
    p_serve: float,
    p_return: float,
) -> Dict[Tuple[int, int], float]:
    d_total: Dict[Tuple[int, int], float] = defaultdict(float)
    s1 = set_score_distribution(set_specs[0], first_set_server_a, p_serve, p_return)
    for (g1a, g1b, s2_opener), p1 in s1.items():
        a_won_s1 = g1a > g1b
        s2 = set_score_distribution(set_specs[1], s2_opener, p_serve, p_return)
        for (g2a, g2b, s3_opener), p2 in s2.items():
            p12 = p1 * p2
            sa = int(a_won_s1) + int(g2a > g2b)
            sb = int(not a_won_s1) + int(g2b > g2a)
            if sa == 2 or sb == 2:
                d_total[(sa, sb)] += p12
                continue
            s3 = set_score_distribution(set_specs[2], s3_opener, p_serve, p_return)
            for (g3a, g3b, _), p3 in s3.items():
                sa3 = sa + int(g3a > g3b)
                sb3 = sb + int(g3b > g3a)
                d_total[(sa3, sb3)] += p12 * p3
    return dict(d_total)


def _match_bo5_variable_last(
    early: SetSpec,
    last: SetSpec,
    first_set_server_a: bool,
    p_serve: float,
    p_return: float,
) -> Dict[Tuple[int, int], float]:
    memo: Dict[Tuple[int, int, bool], Dict[Tuple[int, int], float]] = {}

    def rec(sa: int, sb: int, opener_is_a: bool) -> Dict[Tuple[int, int], float]:
        if sa >= 3 or sb >= 3:
            return {(sa, sb): 1.0}
        key = (sa, sb, opener_is_a)
        if key in memo:
            return memo[key]
        spec = last if sa + sb == 4 else early
        agg: Dict[Tuple[int, int], float] = defaultdict(float)
        for (ga, gb, next_opener), p in set_score_distribution(spec, opener_is_a, p_serve, p_return).items():
            na = sa + int(ga > gb)
            nb = sb + int(gb > ga)
            sub = rec(na, nb, next_opener)
            for k, v in sub.items():
                agg[k] += p * v
        out = dict(agg)
        memo[key] = out
        return out

    return rec(0, 0, first_set_server_a)


def _match_bo5_uniform(
    spec: SetSpec,
    first_set_server_a: bool,
    p_serve: float,
    p_return: float,
) -> Dict[Tuple[int, int], float]:
    memo: Dict[Tuple[int, int, bool], Dict[Tuple[int, int], float]] = {}

    def rec(sa: int, sb: int, opener_is_a: bool) -> Dict[Tuple[int, int], float]:
        if sa >= 3 or sb >= 3:
            return {(sa, sb): 1.0}
        key = (sa, sb, opener_is_a)
        if key in memo:
            return memo[key]
        agg: Dict[Tuple[int, int], float] = defaultdict(float)
        for (ga, gb, next_opener), p in set_score_distribution(spec, opener_is_a, p_serve, p_return).items():
            na = sa + int(ga > gb)
            nb = sb + int(gb > ga)
            sub = rec(na, nb, next_opener)
            for k, v in sub.items():
                agg[k] += p * v
        out = dict(agg)
        memo[key] = out
        return out

    return rec(0, 0, first_set_server_a)


def _match_win_equation(dist: Dict[Tuple[int, int], float], *, for_a: bool, bo5: bool) -> str:
    """
    One cell: ``84.4% = 35.0% (3–0) + 31.0% (3–1) + …`` with terms ordered from
    most dominant win to least (fewer sets dropped first). Player A cells use
    A–B set counts; Player B cells use B–A so scores read from that player's view.
    """
    if for_a:
        order = [(3, 0), (3, 1), (3, 2)] if bo5 else [(2, 0), (2, 1)]
        p_total = sum(p for (sa, sb), p in dist.items() if sa > sb)
    else:
        order = [(0, 3), (1, 3), (2, 3)] if bo5 else [(0, 2), (1, 2)]
        p_total = sum(p for (sa, sb), p in dist.items() if sb > sa)
    parts: list[str] = []
    for sa, sb in order:
        p = dist.get((sa, sb), 0.0)
        if p <= 0.0:
            continue
        first, second = (sa, sb) if for_a else (sb, sa)
        parts.append(f"{100.0 * p:.1f}% ({first}{_MATCH_SCORE_SEP}{second})")
    if not parts:
        return f"{100.0 * p_total:.1f}% ="
    return f"{100.0 * p_total:.1f}% = " + " + ".join(parts)


def match_formats_table(p_serve: float, p_return: float, first_set_server_a: bool) -> pd.DataFrame:
    standard = SetSpec(6, 6, 7, False)
    slam5_last = SetSpec(6, 6, 10, False)

    rows_meta = [
        (
            "Grand Slam Men's Singles",
            _match_bo5_variable_last(standard, slam5_last, first_set_server_a, p_serve, p_return),
            True,
        ),
        (
            "Grand Slam Women's Singles",
            _slam_bo3_variable_tb(
                (standard, standard, slam5_last),
                first_set_server_a,
                p_serve,
                p_return,
            ),
            False,
        ),
        (
            "ATP/WTA Singles",
            _match_bo3_three_sets(standard, first_set_server_a, p_serve, p_return),
            False,
        ),
        (
            "ATP/WTA Doubles",
            _match_bo3_mtb10(standard, first_set_server_a, p_serve, p_return),
            False,
        ),
        (
            "Next Gen Fast4",
            _match_bo5_uniform(
                SetSpec(4, 3, 7, True),
                first_set_server_a,
                p_serve,
                p_return,
            ),
            True,
        ),
    ]

    cols = [MATCH_TABLE_ROW_LABEL_COL, "Player A wins", "Player B wins"]
    data = []
    for label, dist, is_bo5 in rows_meta:
        data.append(
            [
                label,
                _match_win_equation(dist, for_a=True, bo5=is_bo5),
                _match_win_equation(dist, for_a=False, bo5=is_bo5),
            ]
        )
    return pd.DataFrame(data, columns=cols)


def clear_caches() -> None:
    _advantage_game_tables.cache_clear()
    _tiebreak_solve.cache_clear()
    _noad_game_rec.cache_clear()
    _noad_deuce_visit_rec.cache_clear()


def advantage_game_win_prob_iid(p: float) -> float:
    """P(A wins standard game) when P(A wins point)=p on every point (A serving throughout)."""
    m, _ = _advantage_game_tables(p)
    return float(sum(m[(0, 0)][i] for i in (4, 5, 6, 7)))


def no_ad_margin_pmf_closed_form(p: float) -> Tuple[float, ...]:
    """
    Closed-form probability of each MARGIN_COLS bucket for no-ad, A always serving, i.i.d. p.
    """
    q = 1.0 - p
    lst = list(_zero_dist())

    # Win 4-0
    lst[_bucket_for_margin(4)] += p**4
    # Win 4-1: C(4,3) p^4 q
    lst[_bucket_for_margin(3)] += 4 * (p**4) * q
    # Win 4-2: C(5,3) p^4 q^2
    lst[_bucket_for_margin(2)] += 10 * (p**4) * q**2
    # Win 4-3: C(6,3) p^4 q^3
    lst[_bucket_for_margin(1)] += 20 * (p**4) * q**3
    # Lose 0-4
    lst[_bucket_for_margin(-4)] += q**4
    # Lose 1-4: C(4,3) q^4 p
    lst[_bucket_for_margin(-3)] += 4 * (q**4) * p
    # Lose 2-4: C(5,3) q^4 p^2
    lst[_bucket_for_margin(-2)] += 10 * (q**4) * p**2
    # Lose 3-4: C(6,3) q^4 p^3
    lst[_bucket_for_margin(-1)] += 20 * (q**4) * p**3

    return tuple(lst)


def advantage_game_win_prob_closed_form_polynomial(p: float) -> float:
    """
    Polynomial form for P(A wins advantage game) with i.i.d. point prob p (A serving).
    Derived from absorbing Markov chain on (a,b) with a,b <= 4 or deuce strip.
    We validate via identity: should equal DP result for all p in [0,1].
    """
    q = 1.0 - p
    d = advantage_win_from_deuce_closed_form(p)
    # Reach deuce: C(6,3) p^3 q^3
    p_deuce_score = 20 * (p**3) * (q**3)
    # Win before deuce (same structure as no-ad but stop before 3-3 deciding)
    win_before = (p**4) * (1.0 + 4 * q + 10 * q**2)
    return win_before + p_deuce_score * d
