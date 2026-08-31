# 1-D Linear Hall MHD Generator

A one-dimensional simulation of a linear Hall MHD generator: a constant-area
channel march that couples Ohm's law, Saha ionization equilibrium, and
electron/primary-gas energy exchange slice by slice. See `Derivation.md` for
the physics.

- `main.py` -- runs the default operating point, prints a performance
  summary, and shows the matplotlib plots.
- `magnetohydrodynamics/` -- the reusable package (gas/seed models, ionization,
  transport, the Hall solver, presets, analysis helpers).
- `ui/` -- an interactive Streamlit UI (sliders, live plots, and a
  single-parameter optimizer).
- `tests/` -- pytest suite, validated against a frozen reference
  implementation (`tests/reference_main.py`).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running the simulation

```bash
uv run python main.py
```

## Running the tests

```bash
uv run pytest
```

## Running the interactive UI

The UI has its own dependencies (Streamlit) kept in an optional `ui` group so
a plain `uv sync` stays lightweight. Install and run it with:

```bash
uv run --extra ui streamlit run ui/app.py
```

This opens a browser tab with sliders for channel geometry and inlet
conditions (each with adjustable bounds via the "⚙️ Bounds" popover under it),
live-updating plots, a full readout of the plasma state at the channel inlet,
a single- and a multi-parameter optimizer, and a tab to compare configurations
side by side. Configurations can be saved/loaded as TOML files from the
sidebar; they land in `ui/saved_configs/` (git-ignored) by default.
