# Demo video tooling

Records the narrated, click-by-click demo of the app and the README screenshots. Not needed to use the tool.

```bash
python -m venv video/.venv
video/.venv/bin/pip install -r video/requirements.txt
video/.venv/bin/python -m playwright install chromium
# Kokoro voice model (Apache-2.0), ~340 MB, not stored in git:
mkdir -p video/models && cd video/models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ../..
make run &                                         # app on http://localhost:9003
video/.venv/bin/python video/make_video_v2.py      # → video/out/agent_report_card_demo_v2.mp4 (+ _silent)
video/.venv/bin/python video/screenshots.py        # → docs/images/*.png
```

- `narration.json`: scenes made of short beats `{"key", "say", "do"}`. `say` is one or two spoken sentences, `do`
  is exactly what happens on screen while they are spoken.
- `make_video_v2.py`: one action function per beat key (`build_actions`). Each beat's audio is synthesised on its
  own (Kokoro `am_michael`, cached in `out/tts/`); the action starts when the beat's words start, and the screen
  holds only for the rest of that beat's audio, with gentle cursor motion over the thing being described. The
  opening (the flagship run already open in Run detail) is prepared silently and trimmed away.
- Outputs in `out/`: `agent_report_card_demo_v2.mp4`, `_silent.mp4`, `.srt` subtitles, `beat_timeline.json` and
  `beat_frames/` (one frame per beat, taken right after its action) for checking that every screen matches its words.
- `--dry` runs every beat's action with short holds and no audio, and exits non-zero if any action failed.
  `--voice` picks another Kokoro voice, `--speed` the speaking rate (default 1.1).
