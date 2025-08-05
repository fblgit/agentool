.DEFAULT_GOAL := all

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install
install: ## Install the package, dependencies, and pre-commit for local development
	pip install -e ".[all,dev,lint]"
	# pre-commit install --install-hooks  # Disabled for now

.PHONY: format
format: ## Format the code
	ruff format
	ruff check --fix --fix-only

.PHONY: lint
lint: ## Lint the code
	ruff format --check
	ruff check

.PHONY: typecheck-pyright
typecheck-pyright:
	@# PYRIGHT_PYTHON_IGNORE_WARNINGS avoids the overhead of making a request to github on every invocation
	PYRIGHT_PYTHON_IGNORE_WARNINGS=1 pyright

.PHONY: typecheck-mypy
typecheck-mypy:
	mypy

.PHONY: typecheck
typecheck: typecheck-pyright ## Run static type checking

.PHONY: typecheck-both  ## Run static type checking with both Pyright and Mypy
typecheck-both: typecheck-pyright typecheck-mypy

.PHONY: test
test: ## Run tests and collect coverage data
	coverage run -m pytest
	@coverage report

.PHONY: test-fast
test-fast: ## Same as test except no coverage and 4x faster depending on hardware
	pytest -n auto --dist=loadgroup

.PHONY: testcov
testcov: test ## Run tests and generate an HTML coverage report
	@echo "building coverage html"
	@coverage html

.PHONY: codespell
codespell: ## Run codespell to check for common misspellings
	codespell

.PHONY: all
all: format lint typecheck testcov ## Run code formatting, linting, static type checks, and tests with coverage report generation

.PHONY: clean
clean: ## Clean up generated files
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .pytype
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: help
help: ## Show this help (usage: make help)
	@echo "Usage: make [recipe]"
	@echo "Recipes:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)