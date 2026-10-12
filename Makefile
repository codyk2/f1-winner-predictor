PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: help setup data features test eda cv freeze test-2025 report app lint

help:
	@echo "make setup      create .venv and install pinned requirements"
	@echo "make data       download every session into data/raw (cached after the first run)"
	@echo "make features   build data/processed/features.parquet"
	@echo "make test       leakage tests and unit tests"
	@echo "make eda        execute notebooks/01_eda.ipynb (2019-2024 only)"
	@echo "make cv         walk-forward validation, tuning and ablations on 2019-2024"
	@echo "make freeze     write the chosen features and parameters to config/model.json"
	@echo "make test-2025  the locked 2025 run. Refuses to run before the v1.0 tag or twice"
	@echo "make report     rebuild tables and figures from saved predictions"
	@echo "make app        Streamlit demo of the 2025 predictions"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

data:
	$(PY) -m f1pred.ingest

features:
	$(PY) -m f1pred.features

test:
	$(PY) -m pytest -q

eda:
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

cv:
	$(PY) -m f1pred.train cv
	$(PY) -m f1pred.evaluate cv

freeze:
	$(PY) -m f1pred.train freeze

# The only target that reads 2025 outcomes. Kept apart so tuning can't trigger it.
test-2025:
	$(PY) -m f1pred.evaluate test-2025

report:
	$(PY) -m f1pred.evaluate report
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/02_results.ipynb

app:
	$(PY) -m streamlit run app.py

lint:
	$(PY) -m ruff check src tests
