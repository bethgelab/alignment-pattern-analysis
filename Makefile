.PHONY: lint mypy ruff fix test

lint:
	pre-commit run --all-files

mypy:
	pre-commit run mypy --all-files

ruff:
	pre-commit run ruff --all-files

fix:
	ruff check --fix

test:
	python -m pytest --cov=multitasking
