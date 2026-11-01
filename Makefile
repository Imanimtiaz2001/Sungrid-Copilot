.PHONY: install test lint eval demo chat serve offline-eval clean

install:            ## create the virtualenv-independent install (run inside your venv)
	pip install -r requirements-dev.txt && pip install -e . --no-deps

test:               ## full test-suite, offline and deterministic (no API key needed)
	pytest

lint:
	ruff check src tests && ruff format --check src tests

eval:               ## golden-set evaluation with your configured LLM/embedder
	sungrid eval

offline-eval:       ## the exact configuration CI gates on
	EMBEDDING_PROVIDER=hashing LLM_PROVIDER=none sungrid eval --strict

demo:
	python scripts/demo.py

chat:
	sungrid chat --trace

serve:
	sungrid serve

clean:
	rm -rf .index .pytest_cache .ruff_cache .hypothesis
