# Known Limitations & Mitigations

Identified during design review, tracked here so each is a deliberate,
documented decision rather than something discovered late. Referenced by
phase so the fix lands with the code that needs it.

| # | Gap | Mitigation | Config knob | Phase |
|---|---|---|---|---|
| 1 | No representative real footage at project start | Validate pipeline mechanics against public aerial datasets (VisDrone/UAVDT); all scoring stays config-driven so tuning against real footage later is a YAML edit, not a rebuild | `scoring.weights` | 2-4 |
| 2 | ByteTrack ID switching double-counts one object as disappear+appear | Ghost-track reconciliation: hold a dropped track briefly, relabel a same-class reappearance within a motion-plausible radius as a continuation (`logical_id`) instead of a new event | `track_continuity` | 3-4 |
| 3 | Motion compensation unstable over featureless terrain (water, haze, uniform ground) | Quality-gate the homography fit (RANSAC inlier count/ratio); below threshold, skip compensation for that frame pair and fall back to YOLO-only signal | `motion_compensation` | 4 |
| 4 | Whisper can produce confident-looking transcript on noise/static | Confidence-gate transcript segments; below-threshold segments kept in transcript.json for audit but excluded from the "confirmed" evidence tier fed to the LLM as fact | `transcript_confidence` | 6-7 |
| 5 | No link between a run's output and the config/code version that produced it | Stamp `run_metadata` (ontology.yaml hash + git commit hash + timestamp) into every evidence.json/summary.json | — (implemented directly, no tunable) | 7-8 |
| 6 | Single peak-frame keyframe caption can miss what actually mattered in an event | 3 keyframes per event (start/peak/end), still sequential VL calls — flat VRAM cost | `keyframe_captioning` | 7 |
| 7 | Streamlit default bind exposes UI beyond the one workstation | Launch with `--server.address=127.0.0.1`, baked into the launch script | — (launch flag) | 9 |
| 8 | Repo/IP boundary with the company not yet confirmed | Ask explicitly; repo stays private until confirmed; no footage/domain data ever committed regardless (already gitignored) | — (process, not code) | ongoing |
| 13 | Wheelhouse built with the wrong running Python silently omits conditional dependencies | `--python-version 310` only controls which wheel tags pip fetches; it does NOT control evaluation of conditional deps like `exceptiongroup; python_version < "3.11"` — that's evaluated against whichever Python actually runs pip. Discovered when `pip install --no-index -r requirements.txt` failed on a missing `exceptiongroup` (a transitive dep of anyio/httpx/huggingface-hub/faster-whisper) because the wheelhouse was built by `python` resolving to 3.14, not 3.10. Fixed by calling `py -3.10 -m pip download` explicitly in `build_wheelhouse.bat`, not plain `python` | `build_wheelhouse.bat` (uses `py -3.10` explicitly) | 1, ongoing |
| 14 | COCO-pretrained YOLO misses small distant airplanes and top-down UAV targets | Integrated SAHI sliding-window tiling (`sahi.enabled: true` in `ontology.yaml`) + fine-tuning workflow (`scripts/train_yolo.py`) for custom/VisDrone weights | `detection.sahi` | 2, ongoing |
| 15 | QuickTime `.MOV` drone footage causes duration/sampling errors in naive readers | Updated `probe.py` for safe float parsing of `N/A` duration strings and FFmpeg container demuxing in `sample.py` | — (implemented in probe/sample) | 1-2 |
cl