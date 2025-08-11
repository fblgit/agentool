# Repository Guidelines

## Project Structure & Modules
- `src/agentool/`: Core runtime (model, manager, registry, injector).
- `src/agentoolkit/`: Built-in toolkits (storage, auth, network, security, workflows).
- `src/agents/`: Higher-level/coordinating agents.
- `src/ui/`: Optional Streamlit UI components and helpers.
- `src/templates/`: Jinja templates (prompts, skeletons, system).
- `tests/`: Pytest suite; fixtures in `tests/conftest.py`.
- `docs/`: Architecture and guides; some tests run hooks from `docs/.hooks`.
- `examples/`: Usage examples referenced by tests.

## Build, Test, Dev Commands
- Install: `make install` — editable install with extras `[all,dev,lint]`.
- Format: `make format` — ruff format + autofix lint issues.
- Lint: `make lint` — ruff format check and lint.
- Types: `make typecheck` (pyright) or `make typecheck-both` (pyright + mypy).
- Tests: `make test` — pytest via coverage; `make test-fast` for xdist.
- Coverage HTML: `make testcov` — writes `htmlcov/`.
- Misc: `make codespell`, `make clean`, `make help`, `make all`.

## Coding Style & Naming
- Python 3.8+ with type hints required on public APIs.
- Formatting/linting: ruff (configured in `pyproject.toml`). Run `make format` before PRs.
- Indentation: 4 spaces; lines ~100–120 chars per ruff rules.
- Naming: modules/files `snake_case.py`, functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Imports: prefer explicit imports; keep package boundaries clear (`agentool` core vs `agentoolkit`).

## Testing Guidelines
- Framework: pytest with `pytest-xdist`, coverage configured in `pyproject.toml`.
- Location/naming: put tests in `tests/`, files `test_*.py`, functions `test_*`.
- Run examples safely; VCR data lives under `tests/cassettes/`.
- Commands: `pytest -n auto`, specific file: `pytest tests/test_agentool.py`.
- Aim for meaningful coverage; HTML report via `make testcov`.

## Commit & PR Guidelines
- Commits: concise, imperative subject; include scope where helpful (e.g., `[ui]`, `[registry]`). Optional Conventional Commits prefixes (`feat:`, `fix:`, `chore:`) are welcome.
- PRs: include a clear description, linked issues, test plan/results, and screenshots for UI changes. Update docs/examples if behavior changes.
- CI expectations: passes format/lint/typecheck/tests; no new flake/ruff errors.

## Security & Configuration
- Keep secrets out of the repo; prefer environment variables in examples/tests.
- Optional extras are declared in `pyproject.toml` (e.g., `[storage, auth, http, ui]`). Install only what you need.
- Avoid committing generated artifacts (`.coverage`, `htmlcov/`, `logs/`, `tmp/`). Use `make clean` to reset.

