# Game, Set, Match

Interactive Streamlit app exploring how serve/return point probabilities propagate into games, tiebreaks, and common match formats.

## Setup

```bash
cd /path/to/game-set-match
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
streamlit run app.py
```

## Model notes

- Point outcomes are i.i.d. given the server (`p_serve` / `p_return` for Player A).
- The first table fixes **Player A as server** for deuce/no-ad games and **A serves point 1** of each standalone tiebreak row so columns stay comparable.
- Margin columns include **±1** (e.g. 7–6 tiebreaks, no-ad 4–3) so point-level outcomes partition to 100%.
- If **p_serve + p_return = 1**, alternating serve makes the match **symmetric**: set and match win rates for A are **exactly 50%** (not a bug). Move the sliders so the sum differs from 1 to favor one player.
