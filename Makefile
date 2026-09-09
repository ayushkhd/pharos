# Developer entry points. Every target runs inside the uv-managed virtualenv.
#
#   make install    create .venv with all extras and dev tools
#   make fmt        rewrite files to the house style (isort + pyink)
#   make lint       fail on style drift or pylint findings
#   make typecheck  mypy over the package
#   make test       pytest
#   make check      everything CI would run

.PHONY: install fmt lint typecheck test check

PY_DIRS := pharos tests

install:
	uv sync --all-extras

fmt:
	uv run isort $(PY_DIRS)
	uv run pyink $(PY_DIRS)

lint:
	uv run isort --check-only --diff $(PY_DIRS)
	uv run pyink --check --diff $(PY_DIRS)
	uv run pylint $(PY_DIRS)

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
