.PHONY: help install add add-dev remove test lint format clean build

help:
	@echo "Available recipes:"
	@echo "  install     Install all dependencies (first time or after pulling changes)"
	@echo "  add         Add a package - usage: make add PKG=package_name"
	@echo "  add-dev     Add a dev dependency - usage: make add-dev PKG=package_name"
	@echo "  remove      Remove a package - usage: make remove PKG=package_name"
	@echo "  test        Run tests with pytest"
	@echo "  lint        Lint and check code formatting with ruff"
	@echo "  format      Format code in place with ruff"
	@echo "  clean       Remove build artifacts and cache directories"
	@echo "  build       Build wheel distribution"

install:
	@poetry install

add:
	@poetry add $(PKG)

add-dev:
	@poetry add --group dev $(PKG)

remove:
	@poetry remove $(PKG)

test:
	@poetry run pytest tests/ -v

format:
	@poetry run ruff format unbiastap tests

fix:
	@poetry run ruff check --fix unbiastap/

fmt: format fix

lint:
	@poetry run ruff check unbiastap tests

ready: fmt lint test

clean:
	@rm -rf dist/ .mypy_cache/ .ruff_cache/ .pytest_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} +