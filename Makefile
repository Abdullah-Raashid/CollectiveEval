.PHONY: test lint typecheck

test:
	@if python3 -c "import pytest" >/dev/null 2>&1; then \
		PYTHONPATH=src python3 -m pytest tests -q; \
	else \
		PYTHONPATH=src python3 -m unittest discover -s tests -v; \
	fi

lint:
	python3 -m ruff check src tests

typecheck:
	python3 -m mypy src
