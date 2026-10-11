PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test data smoke pilot check paper
test:
	$(PYTHON) -m unittest discover -s tests

data:
	$(PYTHON) -m security_steering build-data data/processed/controlled.jsonl

smoke: data
	$(PYTHON) -m security_steering run --config configs/smoke.yaml

pilot:
	$(PYTHON) -m security_steering build-data data/processed/pilot.jsonl --classes override source_confusion
	$(PYTHON) -m security_steering run --config configs/cpu_pilot.yaml

check:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m ruff format --check src scripts tests

paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
