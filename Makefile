# Makefile for Secure Smart EV Charging Research Framework
# Enforces reproducible execution across tests, baseline, attack, and mitigation experiments.

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest

.PHONY: all setup test baseline attack detect headline energy-check clean

all: test headline

setup:
	python3 -m venv "$(VENV)"
	"$(PIP)" install --upgrade pip
	"$(PIP)" install -e ".[dev]"

test:
	PYTHONPATH="src:." "$(PYTEST)" -v

baseline:
	PYTHONPATH="src:." "$(PYTHON)" experiments/run_baseline.py
	PYTHONPATH="src:." "$(PYTHON)" experiments/compare_baseline_vs_mpc.py

attack:
	PYTHONPATH="src:." "$(PYTHON)" experiments/run_attack_and_detection_experiments.py

detect:
	PYTHONPATH="src:." "$(PYTHON)" experiments/run_attack_and_detection_experiments.py

headline:
	PYTHONPATH="src:." "$(PYTHON)" experiments/run_headline_experiment.py

energy-check:
	PYTHONPATH="src:." "$(PYTHON)" experiments/verify_energy_conservation.py

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
