# AirSense V2 — one-command entry points.
#
# Release engineering only. Nothing here retrains, rescores or regenerates any
# scientific artifact. The frozen evidence chain is never rebuilt by a Makefile.

PY      := python3
VENV    := .venv-v2/bin/python
UV      := uv

.DEFAULT_GOAL := help
.PHONY: help env data test test-fast validate validate-fast validate-metadata check clean-pyc

help:                     ## Show this help
	@echo "AirSense V2 — available targets"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Quick start:  make env && make data && make check"
	@echo "Note: three validators fail BY DESIGN. 'make validate' explains which."

env:                      ## Reconstruct the frozen study environment (.venv-v2)
	$(UV) venv --python 3.11 .venv-v2
	$(UV) pip install --python $(VENV) -r requirements-v2-core-lock.txt
	$(UV) pip install --python $(VENV) torch==2.14.0 \
	    --index-url https://download.pytorch.org/whl/cpu
	$(UV) pip install --python $(VENV) safetensors==0.8.0
	@echo
	@echo "Figure code additionally needs Pillow, whose historical version was"
	@echo "never recorded — see docs/PHASE_12_REPRODUCIBILITY_NOTES.md."
	@echo "  uv pip install --python $(VENV) Pillow"

data:                     ## Rebuild the derived Phase-2 layer (gitignored, ~256 MB)
	@echo "Rebuilding data/processed from tracked data/raw ..."
	$(VENV) scripts/build_forecast_dataset.py
	@echo "Every output digest is recorded in artifacts/v2_phase2_manifest.json."

test:                     ## Full unit suite (needs 'make data' first)
	$(VENV) -m unittest discover -s tests

test-fast:                ## Unit tests that need no derived data layer (needs 'make env')
	@# Uses .venv-v2 because test_itransformer imports torch, which the system
	@# interpreter does not carry. "fast" here means "no data/processed", not
	@# "no environment".
	$(VENV) -m unittest tests.test_preprocessing tests.test_target_history \
	    tests.test_metrics tests.test_b3_features tests.test_scoring \
	    tests.test_itransformer

validate:                 ## Every validator, with expected failures annotated
	$(PY) scripts/run_all_validators.py

validate-fast:            ## Skip validators that re-run the whole upstream chain
	$(PY) scripts/run_all_validators.py --fast

validate-metadata:        ## Only validators needing no derived data layer
	$(PY) scripts/run_all_validators.py --metadata-only

check: test validate      ## Unit suite + full validator sweep

clean-pyc:                ## Remove __pycache__ (never touches artifacts)
	find . -path ./.git -prune -o -name '__pycache__' -type d -print0 \
	  | xargs -0 -r rm -rf
