PY ?= python
PORT ?= 9003

.PHONY: help install dev run test analyze demo samples examples train
help:      ## list commands
	@grep -E "^[a-z]+:.*##" Makefile | sed -E "s/:.*## /\t/"
install:   ## install runtime dependencies
	$(PY) -m pip install -r requirements.txt
dev:       ## install runtime + test dependencies
	$(PY) -m pip install -r requirements-dev.txt
run:       ## start the app on http://localhost:$(PORT)
	$(PY) -m streamlit run app.py --server.port $(PORT)
test:      ## run the test suite
	$(PY) -m pytest -q tests
analyze:   ## CLI demo: analyse the demo log with the SkyDesk policy
	$(PY) -m arc analyze sample_logs/demo_agent_runs.log --policy sample_logs/skydesk_policy.yaml
demo:      ## regenerate the curated demo set in sample_logs/ (deterministic)
	$(PY) scripts/generate_sample_logs.py && $(PY) scripts/validate_sample_logs.py
samples:   ## generate the bulk sets (300 SkyDesk runs in every format + ShopDesk training) into build/samples/
	$(PY) scripts/generate_sample_logs.py --only bulk && $(PY) scripts/validate_sample_logs.py build/samples
examples:  ## regenerate examples/ from the demo set
	$(PY) scripts/make_examples.py
train:     ## retrain the success model (generates the ShopDesk training set into build/samples/ if needed)
	$(PY) scripts/train_success_model.py
