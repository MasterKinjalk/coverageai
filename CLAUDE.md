# CoverageVision AI: notes for Claude Code

- Setup and commands: `README.md`. Open work: `Docs/NEXT_STEPS.md` (start there).
- Scope and success criteria: `Docs/CoverageVisionAI_Project_Proposal (1).pdf` (Table 1). Design decisions: `Docs/decisions.md` (add a row for every new design choice). Original work-package plan: `PLAN.md`.
- Run everything through uv: `uv run pytest -q`, `uv run streamlit run app/main.py`. If CUDA torch was installed with `uv pip`, use `uv run --no-sync` so the CPU torch from the lock file is not restored.
- New controller features stay opt-in and default off; with them off, existing results and tests must be bit-identical (check the simulator's RNG draw-order contract in its docstring before touching `simulator.py`).
- Evaluations: paired comparisons on the same seeds, paired-bootstrap 95% CIs, fair oracle (`oracle`) as the main reference, `oracle_card` only as an unreachable bound. Report negative results as such.
- Append one line per session to `Docs/ai_usage_log.md` (date, task, model, what it produced).
