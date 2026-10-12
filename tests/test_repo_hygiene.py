"""Guards for the project's own-data rule: every dataset, doc and narration line is ours."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "build"}   # build/ = generated, git-ignored
SKIP_PREFIXES = ("video/out/", "video/models/")   # recordings and TTS weights (regenerated / binary)
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".log", ".csv", ".txt", ".toml", ".ipynb", ".cfg",
            ".ini", ".sh", ".html", ""}   # "" = dotfiles such as .gitignore
# names of outside datasets / their publisher that must never appear (split so this file does not match itself)
BANNED = re.compile("har" "bou?r|deploy" "ment|op" "03", re.IGNORECASE)


def _files():
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]) or rel.as_posix().startswith(SKIP_PREFIXES) \
                or not p.is_file():
            continue
        yield rel, p


def test_no_outside_dataset_references_in_names_or_text():
    hits = []
    for rel, p in _files():
        if BANNED.search(str(rel)):
            hits.append(f"file name: {rel}")
        if p.suffix.lower() in TEXT_EXT or p.name.startswith("."):
            text = p.read_text(encoding="utf-8", errors="replace")
            if BANNED.search(text):
                hits.append(str(rel))
    assert not hits, hits


def test_bundled_model_is_trained_on_our_own_data():
    m = json.loads((ROOT / "arc" / "models" / "success_model.json").read_text(encoding="utf-8"))
    prov = m["provenance"]
    assert "ShopDesk" in prov["source"] and "synthetic" in prov["source"]
    assert prov["source_file"] == "build/samples/training/shopdesk_train_arl.jsonl"   # generated on demand


def test_narration_scenes_and_numbers():
    """video/narration.json: scenes of short beats {key, say, do}; spoken words only; numbers as the app shows them."""
    scenes = json.loads((ROOT / "video" / "narration.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in scenes] == ["hook", "step1", "step2", "step3", "summary", "actions", "detail", "human",
                                         "accuracy", "build", "close"]
    beats = [b for s in scenes for b in s["beats"]]
    keys = [b["key"] for b in beats]
    assert len(keys) == len(set(keys)), "beat keys must be unique (they map to action functions)"
    for b in beats:
        assert set(b) == {"key", "say", "do"}, b["key"]
        assert 3 <= len(b["say"].split()) <= 26, f"beat {b['key']} is too long for one screen action"
    say = " ".join(b["say"] for b in beats)
    assert 560 <= len(say.split()) <= 800   # about four and a half minutes at the recording speed
    assert not re.search(r"[%$#@&*=<>/\\\d]", say), "narration must be spoken words only"
    assert beats[0]["say"].startswith("My agent told this customer"), "the video opens on the flagship run"
    for phrase in ["failed three times", "what should a policy look like", "Langfuse", "thirty one percent",
                   "eleven action items", "sixteen runs, three hundred thirty six events", "Seven runs claimed success",
                   "supervisor", "zero point nine nine one", "don't trust done", "Built with Claude Code", "synthetic",
                   "With Claude Code", "not a web page"]:
        assert phrase in say, phrase
