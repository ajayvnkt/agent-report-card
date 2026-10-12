"""Shared fixtures.

The repo ships only the curated 16-run demo set (sample_logs/) and the 3-run examples (examples/).
The bulk sets -- 300 SkyDesk runs in every upload format and the 600-run ShopDesk training set --
are generated once per test session into a temporary folder by the same deterministic generator
that `make samples` uses (about half a second).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LOGS = ROOT / "sample_logs"
DEMO_ARL = SAMPLE_LOGS / "demo_agent_runs.jsonl"
DEMO_LOG = SAMPLE_LOGS / "demo_agent_runs.log"
SKY_POLICY = SAMPLE_LOGS / "skydesk_policy.yaml"
SHOP_POLICY = SAMPLE_LOGS / "shopdesk_policy.yaml"
FLAGSHIP_RUN = "sky-0904-0028"     # update_contact failed 3x, the reply says the number was updated


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_sample_logs", ROOT / "scripts/generate_sample_logs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def generator():
    return load_generator()


@pytest.fixture(scope="session")
def bulk_samples(tmp_path_factory, generator) -> Path:
    """Folder with the bulk SkyDesk files (skydesk_arl.jsonl, ..._otlp.json, ..._events.csv, ...)
    and training/shopdesk_train_arl.jsonl, exactly as `make samples` writes them."""
    out = tmp_path_factory.mktemp("samples")
    runs = generator.simulate_skydesk(7, 150)
    generator.write_skydesk_bulk(out, seed=7, runs_per_version=150, otlp_runs=60, runs=runs)
    generator.write_shopdesk(out, seed=11, n_runs=600)
    return out
