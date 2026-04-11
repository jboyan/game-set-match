# Streamlit entrypoint: add local src/ to import path before other imports.
from __future__ import annotations

import html
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import tennis_model as tm

_SLIDER_MAX_W = "450px"
# Streamlit 1.41+ (see Slider.tsx renderInnerTrack): StyledThumbWrapper holds the thumb(s),
# then UIStyledInnerTrack is the *next sibling* — that node is the real 2px Base Web bar.
# Target it with an adjacent-sibling selector (more reliable than deep :has() chains).
_SLIDER_TRACK_BAR_SUFFIX = '[data-baseweb="slider"] div:has(> [role="slider"]) + div'
_SLIDER_TRACK_BAR = f'[data-testid="stSlider"] {_SLIDER_TRACK_BAR_SUFFIX}'
# Prefix match avoids apostrophe / exact-label mismatches between Python and the DOM.
_ARIA_A = "Player A"
_ARIA_B = "Player B"

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
[data-testid="stSlider"] [data-baseweb="slider"] {{
  background: transparent !important;
}}
/* Base Web: outer Track is padded + solid fill; InnerTrack is ~2px with a gradient. Global polish only; colors come from injected gradient CSS. */
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {{
  background: transparent !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}}
/* The padded flex row (Track) keeps theme sliderTrackFill behind InnerTrack — reads as a thin full-width tint on the bar */
[data-testid="stSlider"] [data-baseweb="slider"] div:has(> div:has(> [role="slider"])) {{
  background: transparent !important;
  background-color: transparent !important;
}}
/* Real InnerTrack (last sibling of thumb row in Streamlit's DOM) */
{_SLIDER_TRACK_BAR} {{
  height: 12px !important;
  border-radius: 999px !important;
  align-self: center !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  overflow: hidden !important;
  background-color: transparent !important;
  background-image: none !important;
}}
{_SLIDER_TRACK_BAR}::before,
{_SLIDER_TRACK_BAR}::after {{
  display: none !important;
  content: none !important;
}}
/* Thumb strip stays visually thin; keep it transparent so only the bar above shows color */
[data-testid="stSlider"] [data-baseweb="slider"] div:has(> [role="slider"]) {{
  background: transparent !important;
  box-shadow: none !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
  width: 18px !important;
  height: 18px !important;
  border: 2px solid #111827 !important;
  background: #ffffff !important;
  box-shadow: none !important;
  z-index: 2 !important;
}}
/* Live value while dragging/hovering (Streamlit ThumbValue); keep readable over the track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] > div {{
  color: #111827 !important;
  font-weight: 600 !important;
  background: rgba(255, 255, 255, 0.92) !important;
  padding: 0.15rem 0.45rem !important;
  border-radius: 6px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.14) !important;
}}
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
table.gsm-match-table thead tr {{
  background-color: #f0f2f6;
}}
table.gsm-match-table thead th {{
  font-weight: 600;
  color: rgba(49, 51, 63, 0.85);
}}
table.gsm-match-table td.gsm-match-equation strong {{
  font-weight: 700;
  font-size: 1.15rem;
}}
table.gsm-match-table td.gsm-pct-cell,
table.gsm-match-table th.gsm-pct-cell {{
  text-align: right;
}}
</style>
"""


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


def _game_primitives_table_html(df: pd.DataFrame) -> str:
    """Like the match table: bold leading total in win columns; gray mid header row."""
    mid = tm.GAME_PRIMITIVES_MID_HEADER_ROW_INDEX
    cols = list(df.columns)
    eq_cols = frozenset({"Player A wins", "Player B wins"})
    deuce_col = tm.GAME_DEUCE_COL_HEADER
    thead = "".join(
        f'<th class="gsm-pct-cell">{html.escape(str(c))}</th>'
        if c == deuce_col
        else f"<th>{html.escape(str(c))}</th>"
        for c in cols
    )
    rows_html: list[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        gray = i == mid
        tr_attr = ' style="background-color:#f0f2f6"' if gray else ""
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if c == cols[0]:
                cells.append(f"<td>{html.escape(str(v) if v is not None and not pd.isna(v) else '')}</td>")
            elif c in eq_cols:
                if gray and (v is None or (isinstance(v, str) and not v.strip()) or pd.isna(v)):
                    cells.append("<td></td>")
                else:
                    cells.append(f'<td class="gsm-match-equation">{_match_equation_cell_html(v)}</td>')
            else:
                if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v) and not pd.isna(v):
                    cells.append(
                        f'<td class="gsm-pct-cell">{html.escape(f"{100.0 * float(v):.1f}%")}</td>'
                    )
                else:
                    cells.append(
                        f'<td class="gsm-pct-cell">{html.escape(str(v) if v is not None else "")}</td>'
                    )
        rows_html.append(f"<tr{tr_attr}>" + "".join(cells) + "</tr>")
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
    """Styled native slider with a single red/blue track."""
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    val = int(st.session_state[state_key])
    left, right = st.columns([1.7, 4.3], vertical_alignment="center")
    with left:
        st.markdown(
            f"<p style='margin:0;line-height:1.2;'><strong>{title}: {val}%</strong></p>",
            unsafe_allow_html=True,
        )
    with right:
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


def _match_rows_meta(
    p_serve: float, p_return: float, first_set_server_a: bool
) -> list[tuple[str, dict[tuple[int, int], float], bool]]:
    """Mirror match table formats, plus raw scoreline distributions for charting."""
    standard = tm.SetSpec(6, 6, 7, False)
    slam5_last = tm.SetSpec(6, 6, 10, False)
    return [
        (
            "Grand Slam Men's Singles",
            tm._match_bo5_variable_last(standard, slam5_last, first_set_server_a, p_serve, p_return),
            True,
        ),
        (
            "Grand Slam Women's Singles",
            tm._slam_bo3_variable_tb(
                (standard, standard, slam5_last),
                first_set_server_a,
                p_serve,
                p_return,
            ),
            False,
        ),
        (
            "ATP/WTA Singles",
            tm._match_bo3_three_sets(standard, first_set_server_a, p_serve, p_return),
            False,
        ),
        (
            "ATP/WTA Doubles",
            tm._match_bo3_mtb10(standard, first_set_server_a, p_serve, p_return),
            False,
        ),
        (
            "Next Gen Fast4",
            tm._match_bo5_uniform(
                tm.SetSpec(4, 3, 7, True),
                first_set_server_a,
                p_serve,
                p_return,
            ),
            True,
        ),
    ]


def _match_stack_df(
    p_serve: float, p_return: float, first_set_server_a: bool
) -> pd.DataFrame:
    """One row per stacked segment for a normalized horizontal chart."""
    recs: list[dict[str, object]] = []
    for label, dist, is_bo5 in _match_rows_meta(p_serve, p_return, first_set_server_a):
        if is_bo5:
            recs.extend(
                [
                    {"Format": label, "Segment": "Player A 3-0", "Score": "3-0", "Probability": dist.get((3, 0), 0.0), "Order": 0},
                    {"Format": label, "Segment": "Player A 3-1", "Score": "3-1", "Probability": dist.get((3, 1), 0.0), "Order": 1},
                    {"Format": label, "Segment": "Player A 3-2", "Score": "3-2", "Probability": dist.get((3, 2), 0.0), "Order": 2},
                    {"Format": label, "Segment": "Player A 2-0", "Score": "2-0", "Probability": 0.0, "Order": 0},
                    {"Format": label, "Segment": "Player A 2-1", "Score": "2-1", "Probability": 0.0, "Order": 2},
                    {"Format": label, "Segment": "Player B 2-1", "Score": "2-1", "Probability": 0.0, "Order": 3},
                    {"Format": label, "Segment": "Player B 2-0", "Score": "2-0", "Probability": 0.0, "Order": 5},
                    {"Format": label, "Segment": "Player B 3-2", "Score": "3-2", "Probability": dist.get((2, 3), 0.0), "Order": 3},
                    {"Format": label, "Segment": "Player B 3-1", "Score": "3-1", "Probability": dist.get((1, 3), 0.0), "Order": 4},
                    {"Format": label, "Segment": "Player B 3-0", "Score": "3-0", "Probability": dist.get((0, 3), 0.0), "Order": 5},
                ]
            )
        else:
            recs.extend(
                [
                    {"Format": label, "Segment": "Player A 3-0", "Score": "3-0", "Probability": 0.0, "Order": 0},
                    {"Format": label, "Segment": "Player A 3-1", "Score": "3-1", "Probability": 0.0, "Order": 1},
                    {"Format": label, "Segment": "Player A 3-2", "Score": "3-2", "Probability": 0.0, "Order": 2},
                    {"Format": label, "Segment": "Player A 2-0", "Score": "2-0", "Probability": dist.get((2, 0), 0.0), "Order": 0},
                    {"Format": label, "Segment": "Player A 2-1", "Score": "2-1", "Probability": dist.get((2, 1), 0.0), "Order": 2},
                    {"Format": label, "Segment": "Player B 2-1", "Score": "2-1", "Probability": dist.get((1, 2), 0.0), "Order": 3},
                    {"Format": label, "Segment": "Player B 2-0", "Score": "2-0", "Probability": dist.get((0, 2), 0.0), "Order": 5},
                    {"Format": label, "Segment": "Player B 3-2", "Score": "3-2", "Probability": 0.0, "Order": 3},
                    {"Format": label, "Segment": "Player B 3-1", "Score": "3-1", "Probability": 0.0, "Order": 4},
                    {"Format": label, "Segment": "Player B 3-0", "Score": "3-0", "Probability": 0.0, "Order": 5},
                ]
            )
    out = pd.DataFrame.from_records(recs)
    if out.empty:
        return out
    out["Probability"] = out["Probability"].astype(float)
    out["Order"] = out["Order"].astype(int)
    out["Start"] = 0.0
    for fmt in out["Format"].unique():
        m = out["Format"] == fmt
        ordered = out.loc[m].sort_values(["Order", "Segment"]).index
        p = out.loc[ordered, "Probability"]
        out.loc[ordered, "Start"] = p.cumsum() - p
    out["Start"] = out["Start"].astype(float)
    return out


def _game_stack_df(p_serve: float, p_return: float) -> pd.DataFrame:
    """One row per game/tiebreak equation term for normalized stacked bars."""
    recs: list[dict[str, object]] = []

    def add_terms(row: str, side: str, vals: list[tuple[str, float]], offset: int) -> None:
        for i, (tag, prob) in enumerate(vals):
            recs.append(
                {
                    "Row": row,
                    "Segment": f"Player {side} {tag}",
                    "Label": tag,
                    "Probability": float(prob),
                    "Order": offset + i,
                }
            )

    m_ad, _ = tm._advantage_game_tables(p_serve)
    d_ad = m_ad[(0, 0)]
    sa = tm._advantage_margin2_split_a(p_serve)
    sb = tm._advantage_margin2_split_b(p_serve)
    add_terms(
        "A serves (with deuces)",
        "A",
        [("(@40-0)", d_ad[7]), ("(@40-15)", d_ad[6]), ("(@40-30)", sa[0]), ("(@Ad-in)", sa[1]), ("(@Deuce)", d_ad[4])],
        0,
    )
    add_terms(
        "A serves (with deuces)",
        "B",
        [("(@40-0)", d_ad[0]), ("(@40-15)", d_ad[1]), ("(@40-30)", sb[0]), ("(@Ad-in)", sb[1]), ("(@Deuce)", d_ad[3])],
        5,
    )

    d_na = tm._noad_game_rec(0, 0, p_serve)
    add_terms(
        "A serves (no-ad scoring)",
        "A",
        [("(@40-0)", d_na[7]), ("(@40-15)", d_na[6]), ("(@40-30)", d_na[5]), ("(@Deuce)", d_na[4])],
        0,
    )
    add_terms(
        "A serves (no-ad scoring)",
        "B",
        [("(@40-0)", d_na[0]), ("(@40-15)", d_na[1]), ("(@40-30)", d_na[2]), ("(@Deuce)", d_na[3])],
        5,
    )

    m_bd, _ = tm._advantage_game_tables(p_return)
    d_bd = m_bd[(0, 0)]
    sba = tm._advantage_margin2_split_a(p_return)
    sbb = tm._advantage_margin2_split_b(p_return)
    add_terms(
        "B serves (with deuces)",
        "A",
        [("(@40-0)", d_bd[7]), ("(@40-15)", d_bd[6]), ("(@40-30)", sba[0]), ("(@Ad-in)", sba[1]), ("(@Deuce)", d_bd[4])],
        0,
    )
    add_terms(
        "B serves (with deuces)",
        "B",
        [("(@40-0)", d_bd[0]), ("(@40-15)", d_bd[1]), ("(@40-30)", sbb[0]), ("(@Ad-in)", sbb[1]), ("(@Deuce)", d_bd[3])],
        5,
    )

    d_nb = tm._noad_game_rec(0, 0, p_return)
    add_terms(
        "B serves (no-ad scoring)",
        "A",
        [("(@40-0)", d_nb[7]), ("(@40-15)", d_nb[6]), ("(@40-30)", d_nb[5]), ("(@Deuce)", d_nb[4])],
        0,
    )
    add_terms(
        "B serves (no-ad scoring)",
        "B",
        [("(@40-0)", d_nb[0]), ("(@40-15)", d_nb[1]), ("(@40-30)", d_nb[2]), ("(@Deuce)", d_nb[3])],
        5,
    )

    for target, row in ((7, "Tiebreak to 7"), (10, "Tiebreak to 10")):
        dist, _ = tm._tiebreak_solve(target, target - 1, True, p_serve, p_return)
        add_terms(
            row,
            "A",
            [("(win by 1)", dist[4]), ("(win by 2)", dist[5]), ("(win by 3)", dist[6]), ("(win by 4+)", dist[7])],
            0,
        )
        add_terms(
            row,
            "B",
            [("(win by 1)", dist[3]), ("(win by 2)", dist[2]), ("(win by 3)", dist[1]), ("(win by 4+)", dist[0])],
            5,
        )
    return pd.DataFrame.from_records(recs)


st.set_page_config(page_title="Game, Set, Match", layout="wide")
st.markdown(_APP_CUSTOM_CSS, unsafe_allow_html=True)

st.title("🎾 Game, Set, Match")
_tag_pa = int(st.session_state.get("svc_pct_a", 55))
_tag_pb = int(st.session_state.get("svc_pct_b", 45))
st.markdown(
    f'<p class="gsm-tagline">If a tennis player wins {_tag_pa}% of her points on serve, and her opponent '
    f"wins {_tag_pb}% of his points on serve, what are their chances of winning a game, tiebreak, set, or match?</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
<style>
/* Thick two-tone bar on the real InnerTrack (thumb aria-label distinguishes A vs B). */
[data-testid="stSlider"]:has([aria-label^="{_ARIA_A}"]) {_SLIDER_TRACK_BAR_SUFFIX} {{
  background-color: transparent !important;
  background-image: linear-gradient(to right, #b22222 0%, #b22222 {_tag_pa}%, #2457ae {_tag_pa}%, #2457ae 100%) !important;
  background-repeat: no-repeat !important;
  background-size: 100% 100% !important;
}}
[data-testid="stSlider"]:has([aria-label^="{_ARIA_B}"]) {_SLIDER_TRACK_BAR_SUFFIX} {{
  background-color: transparent !important;
  background-image: linear-gradient(to right, #2457ae 0%, #2457ae {_tag_pb}%, #b22222 {_tag_pb}%, #b22222 100%) !important;
  background-repeat: no-repeat !important;
  background-size: 100% 100% !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

a_pct = _slider_block(
    title="Player A's service point win %",
    state_key="svc_pct_a",
    default=55,
)
b_pct = _slider_block(
    title="Player B's service point win %",
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

game_df = tm.game_primitives_table(p_serve, p_return)
match_df = tm.match_formats_table(p_serve, p_return, True)

st.subheader("Game win probabilities")
game_stack_df = _game_stack_df(p_serve, p_return)
_game_segment_order = [
    "Player A (@40-0)",
    "Player A (win by 4+)",
    "Player A (@40-15)",
    "Player A (win by 3)",
    "Player A (@40-30)",
    "Player A (@Ad-in)",
    "Player A (win by 2)",
    "Player A (@Deuce)",
    "Player A (win by 1)",
    "Player B (win by 1)",
    "Player B (@Deuce)",
    "Player B (win by 2)",
    "Player B (@Ad-in)",
    "Player B (@40-30)",
    "Player B (win by 3)",
    "Player B (@40-15)",
    "Player B (win by 4+)",
    "Player B (@40-0)",
]
_game_segments = [s for s in _game_segment_order if s in set(game_stack_df["Segment"])]
_game_colors = {
    "Player A (@40-0)": "#8b0000",
    "Player A (@40-15)": "#b22222",
    "Player A (@40-30)": "#cc5a5a",
    "Player A (@Ad-in)": "#e07070",
    "Player A (@Deuce)": "#f28b82",
    "Player A (win by 1)": "#f28b82",
    "Player A (win by 2)": "#e07070",
    "Player A (win by 3)": "#cc5a5a",
    "Player A (win by 4+)": "#8b0000",
    "Player B (@40-0)": "#0b3d91",
    "Player B (@40-15)": "#2457ae",
    "Player B (@40-30)": "#4f7fd6",
    "Player B (@Ad-in)": "#72a1eb",
    "Player B (@Deuce)": "#8ab4f8",
    "Player B (win by 1)": "#8ab4f8",
    "Player B (win by 2)": "#72a1eb",
    "Player B (win by 3)": "#4f7fd6",
    "Player B (win by 4+)": "#0b3d91",
}
_game_row_order = [
    "A serves (with deuces)",
    "A serves (no-ad scoring)",
    "B serves (with deuces)",
    "B serves (no-ad scoring)",
    "Tiebreak to 7",
    "Tiebreak to 10",
]
_game_a_totals = (
    game_stack_df.loc[game_stack_df["Segment"].str.startswith("Player A "), ["Row", "Probability"]]
    .groupby("Row", as_index=True)["Probability"]
    .sum()
    .to_dict()
)
_game_row_labels = {
    r: f"{r}: <b>{100.0 * float(_game_a_totals.get(r, 0.0)):.1f}%</b>" for r in _game_row_order
}
_game_row_order_labeled = [_game_row_labels[r] for r in _game_row_order]
game_fig = go.Figure()
_game_legend_seen = {"A": False, "B": False}
for seg in _game_segments:
    sdf = (
        game_stack_df.loc[game_stack_df["Segment"] == seg, ["Row", "Probability", "Label"]]
        .set_index("Row")
        .reindex(_game_row_order)
        .fillna({"Probability": 0.0, "Label": ""})
    )
    probs = [float(x) for x in sdf["Probability"]]
    labels = [str(s) if float(p) >= 0.04 else "" for s, p in zip(sdf["Label"], probs)]
    side = "A" if seg.startswith("Player A ") else "B"
    legend_name = "Player A wins" if side == "A" else "Player B wins"
    show_legend = not _game_legend_seen[side]
    _game_legend_seen[side] = True
    game_fig.add_bar(
        name=legend_name,
        y=_game_row_order_labeled,
        x=probs,
        orientation="h",
        marker_color=_game_colors.get(seg, "#999999"),
        text=labels,
        textposition="inside",
        insidetextanchor="start",
        textfont={"color": "white", "size": 11},
        legendgroup=legend_name,
        showlegend=show_legend,
        customdata=[seg] * len(probs),
        hovertemplate="%{y}<br>%{customdata}: %{x:.1%}<extra></extra>",
    )
game_fig.update_layout(
    barmode="stack",
    barnorm="fraction",
    height=390,
    margin={"l": 10, "r": 10, "t": 8, "b": 90},
    legend={"orientation": "h", "yanchor": "top", "y": -0.24, "x": 0.0, "title": None},
)
game_fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="")
game_fig.update_yaxes(
    title_text="",
    categoryorder="array",
    categoryarray=_game_row_order_labeled[::-1],
    automargin=True,
    tickfont={"size": 16},
)
st.plotly_chart(game_fig, use_container_width=True, config={"displayModeBar": False})

st.subheader("Match win probabilities")
_match_clean = _match_df_na_to_empty_strings(match_df)
stack_df = _match_stack_df(p_serve, p_return, True)
_segment_domain = [
    "Player A 3-0",
    "Player A 3-1",
    "Player A 3-2",
    "Player A 2-0",
    "Player A 2-1",
    "Player B 2-1",
    "Player B 2-0",
    "Player B 3-2",
    "Player B 3-1",
    "Player B 3-0",
]
_segment_colors = {
    "Player A 3-0": "#8b0000",
    "Player A 3-1": "#cc5a5a",
    "Player A 3-2": "#f28b82",
    "Player A 2-0": "#8b0000",
    "Player A 2-1": "#f28b82",
    "Player B 2-1": "#8ab4f8",
    "Player B 2-0": "#0b3d91",
    "Player B 3-2": "#8ab4f8",
    "Player B 3-1": "#4f7fd6",
    "Player B 3-0": "#0b3d91",
}
_format_order = list(match_df[tm.MATCH_TABLE_ROW_LABEL_COL])
_match_a_totals = (
    stack_df.loc[stack_df["Segment"].str.startswith("Player A "), ["Format", "Probability"]]
    .groupby("Format", as_index=True)["Probability"]
    .sum()
    .to_dict()
)
_format_labels = {
    r: f"{r}: <b>{100.0 * float(_match_a_totals.get(r, 0.0)):.1f}%</b>" for r in _format_order
}
_format_order_labeled = [_format_labels[r] for r in _format_order]
fig = go.Figure()
_match_legend_seen = {"A": False, "B": False}
for seg in _segment_domain:
    sdf = (
        stack_df.loc[stack_df["Segment"] == seg, ["Format", "Probability", "Score"]]
        .set_index("Format")
        .reindex(_format_order)
        .fillna({"Probability": 0.0, "Score": ""})
    )
    probs = [float(x) for x in sdf["Probability"]]
    labels = [str(s) if float(p) >= 0.035 else "" for s, p in zip(sdf["Score"], probs)]
    side = "A" if seg.startswith("Player A ") else "B"
    legend_name = "Player A wins" if side == "A" else "Player B wins"
    show_legend = not _match_legend_seen[side]
    _match_legend_seen[side] = True
    fig.add_bar(
        name=legend_name,
        y=_format_order_labeled,
        x=probs,
        orientation="h",
        marker_color=_segment_colors[seg],
        text=labels,
        textposition="inside",
        insidetextanchor="start",
        textfont={"color": "white"},
        legendgroup=legend_name,
        showlegend=show_legend,
        customdata=[seg] * len(probs),
        hovertemplate="%{y}<br>%{customdata}: %{x:.1%}<extra></extra>",
    )
fig.update_layout(
    barmode="stack",
    barnorm="fraction",
    height=340,
    margin={"l": 10, "r": 10, "t": 10, "b": 60},
    legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0.0, "title": None},
)
fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="")
fig.update_yaxes(
    title_text="",
    categoryorder="array",
    categoryarray=_format_order_labeled[::-1],
    automargin=True,
    tickfont={"size": 16},
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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

- **Prob(deuce)** (game table, last column): for **games**, the chance the score ever reaches **40–40** (deuce) before someone holds. **Tiebreak** rows in that column are still **P(extra pts)** — **6–6** (TB7) or **9–9** (TB10) before someone wins by two — even though the column header says **Prob(deuce)**.
- **Prob(deuce) (games, detail):** 40–40 (3–3 points) before the game ends.
- **Prob(extra pts) (tiebreak to 7):** 6–6 before someone wins by two.
- **Prob(extra pts) (tiebreak to 10):** 9–9 before someone wins by two.
- **Win equations (games):** each term is the chance Player A (or B) wins **via that terminal game score path** (e.g. **(@40-0)**).
- **Win equations (tiebreaks):** terms are **(win by 1)**, **(win by 2)**, **(win by 3)**, **(win by 4+)** — the winner’s **point margin** at the end of the tiebreak (same buckets as the old margin columns; see point-margin note below).
- **Point margins:** signed point differential for Player A at the end of that unit (extra ±1 buckets cover no-ad and tiebreaks).
- **Between sets:** the ATP/WTA continuation rule is used — whoever would have served the next game (had the previous set continued) serves game 1 of the new set. Equivalently, the set-opener flips when the set had an odd number of games (6–1, 6–3, 7–6) and stays the same when it had an even number (6–0, 6–2, 6–4, 7–5).
- **Why 50–50 match odds?** If **Player A’s and Player B’s service point win rates are equal**, then whenever A serves, B’s chance to win the point is the same as A’s when B serves (roles swap cleanly). The whole match is then **symmetric** and Player A’s set/match win probability is **exactly ½**, with mirrored score distributions—not a bug.
- **First server:** The app assumes **Player A serves game 1 of set 1.** Under the ATP/WTA continuation rule, who serves first can affect match win probabilities (unlike the always-flip ITF rule), so the choice of first server is meaningful.
        """
    )

with st.expander("Game win probabilities table"):
    st.markdown(_game_primitives_table_html(game_df), unsafe_allow_html=True)

with st.expander("Match win probabilities table"):
    st.markdown(_match_table_html(_match_clean), unsafe_allow_html=True)
