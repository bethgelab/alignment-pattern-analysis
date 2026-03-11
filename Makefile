.PHONY: lint mypy ruff fix test

lint: ruff mypy

mypy:
	mypy .

ruff:
	ruff check .

fix:
	ruff check --fix

test:
	python -m pytest --cov=multitasking
