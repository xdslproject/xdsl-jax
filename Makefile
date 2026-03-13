MAKEFLAGS += --warn-undefined-variables
SHELL := bash

# use a default prefix for coverage data files
COVERAGE_FILE ?= .coverage

# allow overriding which dependency groups are installed
VENV_GROUPS ?= --group dev --group docs

# set default lit options
LIT_OPTIONS ?= -v --order=smart


.PHONY: install
install: .venv/ pre-commit

.venv/:
	uv sync ${VENV_GROUPS}

.PHONY: pre-commit
pre-commit: .venv/
	uv run pre-commit install

.PHONY: check
check: .venv/
	uv run pre-commit run --all-files

.PHONY: pyright
pyright: .venv/
	uv run pyright $(shell git diff --staged --name-only  -- '*.py')

.PHONY: tests
tests: pytest filecheck

.PHONY: pytest
pytest: .venv/
	uv run pytest -W error --cov

.PHONY: filecheck
filecheck: .venv/
	uv run lit $(LIT_OPTIONS) tests/filecheck

.PHONY: coverage
coverage: coverage-tests coverage-filecheck-tests
	uv run coverage combine --append
	uv run coverage report

.PHONY: coverage-ci
coverage-ci: coverage-tests coverage-filecheck-tests
	uv run coverage combine --append
	uv run coverage report
	uv run coverage xml

.PHONY: coverage-tests
coverage-tests: .venv/
	COVERAGE_FILE="${COVERAGE_FILE}.$@" uv run pytest -W error --cov

.PHONY: coverage-filecheck-tests
coverage-filecheck-tests: .venv/
	COVERAGE_FILE="${COVERAGE_FILE}.$@" uv run lit $(LIT_OPTIONS) tests/filecheck -DCOVERAGE

.PHONY: coverage-clean
coverage-clean: .venv/
	uv run coverage erase

.PHONY: docs
docs: .venv/
	uv run mkdocs serve
	uv run mkdocs build

.PHONY: clean-caches
clean-caches:
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache/ .coverage*
	find . -not -path "./.venv/*" | \
		grep -E "(/__pycache__$$|\.pyc$$|\.pyo$$)" | \
		xargs rm -rf

.PHONY: clean
clean: clean-caches
	rm -rf ${VENV_DIR}
