# Streamlit entrypoint: add local src/ to import path before other imports.
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

import tennis_model as tm

_SLIDER_MAX_W = "450px"
_GAME_AB_WIN_COLS = frozenset({"Player A win %", "Player B win %"})

_APP_CUSTOM_CSS = f"""
<style>
p.gsm-tagline {{
  font-size: 1.22rem;
  line-height: 1.5;
  margin: 0 0 0.85rem 0;
  color: inherit;
  opacity: 0.92;
}}
[data-testid="stSlider"] {{ max-width: {_SLIDER_MAX_W}; }}
/* HTML-table fallback if a Streamlit build ignores Styler text-align */
[data-testid="stDataFrame"] table tbody td:not(:first-child) {{
  text-align: right !important;
}}
/* Match table (markdown HTML): bold only the leading total; body matches app font */
table.gsm-match-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
  margin: 0;
}}
table.gsm-match-table th, table.gsm-match-table td {{
  padding: 0.45rem 0.65rem;
  text-align: left;
  vertical-align: top;
  border-bottom: 1px solid rgba(49, 51, 63, 0.12);
}}
table.gsm-match-table thead th {{
  font-weight: 600;
  color: rgba(49, 51, 63, 0.85);
}}
table.gsm-match-table td.gsm-match-equation strong {{
  font-weight: 700;
  font-size: 1.15rem;
}}
</style>
"""


def _dataframe_column_config(
    columns: pd.Index,
    *,
    left_align: frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Pin the first column (``Match Format`` / ``Single game``);
    right-align other columns unless listed in ``left_align``.
    """
    la = left_align or frozenset()
    cfg: dict[str, dict[str, Any]] = {}
    label_col = columns[0]
    for col in columns:
        name = str(col)
        if col == label_col:
            cfg[name] = {**st.column_config.TextColumn(pinned=True), "alignment": "left"}
        else:
            align = "left" if name in la else "right"
            cfg[name] = {**st.column_config.TextColumn(), "alignment": align}
    return cfg


def _match_df_na_to_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Glide/Arrow often shows nulls as the literal text 'None' even when Styler uses na_rep.
    Empty strings avoid nulls in the payload so cells stay blank.
    """
    out = df.copy()
    row_label = out.columns[0]
    for c in out.columns:
        if c == row_label:
            continue
        out[c] = out[c].where(out[c].notna(), "")
    return out


def _fmt_prob_cell(v: object, *, blank_na: bool = False) -> str:
    miss = "" if blank_na else "—"
    if v == "":
        return miss
    if isinstance(v, str) and v.strip().lower() in ("none", "nan"):
        return miss
    if v is None:
        return miss
    try:
        if pd.isna(v):
            return miss
    except TypeError:
        pass
    if isinstance(v, (float, int)) and isinstance(v, float) and v != v:
        return miss
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{100.0 * float(v):.1f}%"
    return str(v)


def _probability_table_styler(
    df: pd.DataFrame,
    *,
    mid_header_row: int | None = None,
    blank_na: bool = False,
) -> pd.io.formats.style.Styler:
    pct_cols = list(df.columns[1:])
    miss = "" if blank_na else "—"
    ab_win_css = "font-weight: 700; font-size: 1.2rem;"

    def _fmt(v: object) -> str:
        return _fmt_prob_cell(v, blank_na=blank_na)

    def _ab_win_data_cells(row: pd.Series) -> list[str]:
        """Bold Player A/B game win % cells; skip the gray mid-header row (game table)."""
        if mid_header_row is not None and row.name == mid_header_row:
            return [""] * len(row)
        return [ab_win_css if c in _GAME_AB_WIN_COLS else "" for c in row.index]

    styler = (
        df.style.hide(axis="index")
        .format(_fmt, subset=pct_cols, na_rep=miss)
        .apply(_ab_win_data_cells, axis=1)
    )
    if mid_header_row is not None:
        gray = "#f0f2f6"

        def _mid_header_row_style(row: pd.Series) -> list[str]:
            if row.name != mid_header_row:
                return [""] * len(row)
            return [f"background-color: {gray};"] * len(row)

        styler = styler.apply(_mid_header_row_style, axis=1)
    return styler


def _match_equation_cell_html(s: object) -> str:
    """Bold only the leading total (``84.4%``); remainder stays normal weight."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    t = str(s).strip()
    if not t:
        return ""
    esc = html.escape
    if " = " not in t:
        return esc(t)
    head, _, tail = t.partition(" = ")
    return f'<strong>{esc(head)}</strong> = {esc(tail)}'


def _match_table_html(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    thead = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows_html: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if c == cols[0]:
                cells.append(f"<td>{html.escape(str(v) if v is not None else '')}</td>")
            else:
                cells.append(f'<td class="gsm-match-equation">{_match_equation_cell_html(v)}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="gsm-match-table" role="grid">'
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
    )


def _slider_block(
    *,
    title: str,
    state_key: str,
    default: int,
) -> int:
    """Label above the slider (value is only on the widget)."""
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    st.markdown(
        '<p style="margin:0 0 0.35rem 0;color:inherit;line-height:1.35;">'
        f"<strong>{title}</strong></p>",
        unsafe_allow_html=True,
    )
    return int(
        st.slider(
            title,
            0,
            100,
            step=1,
            key=state_key,
            label_visibility="collapsed",
            format="%d%%",
        )
    )


st.set_page_config(page_title="Game, Set, Match", layout="wide")
st.markdown(_APP_CUSTOM_CSS, unsafe_allow_html=True)

st.title("🎾 Game, Set, Match")
st.markdown(
    '<p class="gsm-tagline">If a tennis player wins 55% of her points on serve, what are her chances of '
    "winning a game, set, or match?</p>",
    unsafe_allow_html=True,
)

a_pct = _slider_block(
    title="Player A service point win %",
    state_key="svc_pct_a",
    default=55,
)
b_pct = _slider_block(
    title="Player B service point win %",
    state_key="svc_pct_b",
    default=45,
)
p_serve = a_pct / 100.0
p_b_serve = b_pct / 100.0
p_return = 1.0 - p_b_serve

if a_pct == b_pct:
    st.info(
        "**Player A and B have the same service point win rate.** Under alternating serve, that makes "
        "the process symmetric: **set and match win rates are exactly 50% each**, and set-score lines "
        "mirror (e.g. 25% each for 0–2, 1–2, 2–1, 2–0 in best-of-3). Nudge either slider so the two "
        "percentages differ to see a favorite."
    )

tm.clear_caches()
game_df = tm.game_primitives_table(p_serve, p_return)
match_df = tm.match_formats_table(p_serve, p_return, True)

st.subheader("Game win probabilities")
_game_styled = _probability_table_styler(
    game_df,
    mid_header_row=tm.GAME_PRIMITIVES_MID_HEADER_ROW_INDEX,
)
st.dataframe(
    _game_styled,
    use_container_width=True,
    hide_index=True,
    column_config=_dataframe_column_config(_game_styled.data.columns),
)

st.subheader("Match win probabilities")
_match_clean = _match_df_na_to_empty_strings(match_df)
st.markdown(_match_table_html(_match_clean), unsafe_allow_html=True)

with st.expander("Definitions"):
    st.markdown(
        """
### Match formats

Each row in **Match win probabilities** is a full match under that format’s rules (same service-point inputs everywhere).

- **Grand Slam Men’s Singles:** Best-of-**five** sets. Sets **1–4** are standard six-game sets with a first-to-**7** (win by two) tiebreak at **6–6**. If the match reaches **two sets all**, the **fifth** set uses the same six-game structure but a first-to-**10** (win by two) tiebreak at **6–6** (current men’s major format).
- **Grand Slam Women’s Singles:** Best-of-**three** sets. Sets **1–2** are standard six-game sets with a **7-point** tiebreak at **6–6**. A **deciding third** set uses a first-to-**10** (win by two) tiebreak at **6–6** instead of a “normal” third set (current women’s / equal Slam format).
- **ATP/WTA Singles:** Best-of-three. Every set is a standard six-game set with a **7-point** tiebreak at **6–6** (no match tiebreak).
- **ATP/WTA Doubles:** Best-of-three, but if the sides split the first two sets the match is decided by a **match tiebreak** alone: one first-to-**10** (win by two) tiebreak replaces a full third set (Champions / match tiebreak).
- **Next Gen Fast4:** Best-of-**five** **short** sets: first to **four** games, win by two, with a **7-point** tiebreak at **3–games all**. Games use **no-ad** scoring (deciding point at deuce). Every set in the match follows these rules.

### Terms used elsewhere

- **Prob(deuce) (games):** probability the score ever reaches 40–40 (3–3 points) before the game ends.
- **Prob(extra pts) (tiebreak to 7):** probability the tiebreak score is ever 6–6 before someone wins by two.
- **Prob(extra pts) (tiebreak to 10):** probability the tiebreak score is ever 9–9 before someone wins by two.
- **Point margins:** signed point differential for Player A at the end of that unit (extra ±1 buckets cover no-ad and tiebreaks).
- **Between sets:** the ATP/WTA continuation rule is used — whoever would have served the next game (had the previous set continued) serves game 1 of the new set. Equivalently, the set-opener flips when the set had an odd number of games (6–1, 6–3, 7–6) and stays the same when it had an even number (6–0, 6–2, 6–4, 7–5).
- **Why 50–50 match odds?** If **Player A’s and Player B’s service point win rates are equal**, then whenever A serves, B’s chance to win the point is the same as A’s when B serves (roles swap cleanly). The whole match is then **symmetric** and Player A’s set/match win probability is **exactly ½**, with mirrored score distributions—not a bug.
- **First server:** The app assumes **Player A serves game 1 of set 1.** Under the ATP/WTA continuation rule, who serves first can affect match win probabilities (unlike the always-flip ITF rule), so the choice of first server is meaningful.
        """
    )
