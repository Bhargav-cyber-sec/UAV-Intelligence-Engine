# Offline Multi-Modal UAV Intelligence Engine — Technical Design Review

**Prepared for:** DRDO-guided B.E. capstone project
**Constraints locked in for this review:** NVIDIA Quadro P2000 (4 GB VRAM), no USB transfer to the air-gapped PC (Google Drive / email only), project starts Week 1 (nothing built yet), 10-week Mon–Wed home / Thu–Fri DRDO schedule.

---

## 0. Executive Verdict (read this first)

**Yes, this is achievable in 10 weeks — but only if you cut it down from "research lab" scope to "working pipeline with one good model per stage" scope, and only if you solve the offline-transfer problem in Week 1, not Week 8.**

Two of your assumptions need correcting before anything else:

1. **Qwen3-VL as the primary reasoning engine is the wrong call on a 4 GB card.** A vision-language model doing full multimodal reasoning over dozens of keyframes needs more headroom than a P2000 gives you once YOLO/Whisper have also touched the GPU in the same session. The fix isn't "don't use vision" — it's *where* vision is used. Full explanation in §7.
2. **"Google Drive or email" for offline provisioning is a genuine risk, not a minor logistics detail.** Base models (YOLO weights, Whisper weights, Qwen GGUF files, TTS voices) run from ~50 MB to several GB each. Gmail caps attachments at 25 MB; Drive works but large multi-GB uploads/downloads over a facility network you don't control are slow and failure-prone, and — separately — you should confirm with your guide that pulling files into an air-gapped machine via a cloud service is actually sanctioned procedure there, because "air-gapped" and "receives files via Google Drive" are in tension as a security model. That's not your call to fix technically, but it is a question worth raising explicitly with your guide in Week 1, since if the facility's actual policy is "no external transfer channel of any kind," your whole install strategy needs to be built around an *approved* transfer method (could still be USB with prior authorization, DVD/optical media, or a facility-managed transfer server) instead. I'll design the wheelhouse to be as small as possible regardless of which channel you end up using.

---

## 1. What "Event" Means — Ontology Before Algorithm

Don't let YOLO define your events. YOLO answers "what object is in this frame," not "did something mission-relevant just happen." You need an explicit ontology layer between detection and scoring.

**Three-tier event model:**

| Tier | Definition | Example | Source |
|---|---|---|---|
| **Tier 0 — Observation** | A raw per-frame detection/measurement. Not an event yet. | "car, conf 0.81, bbox (x,y,w,h), frame 4120" | YOLO, frame-diff, VAD |
| **Tier 1 — Candidate Event** | A temporally coherent pattern of observations that crosses a persistence/significance threshold. | "object track #17 (car) persisted 4.2s, entered from left edge" | Event scoring engine (§5) |
| **Tier 2 — Mission Event** | A Tier-1 candidate that is labeled against the domain ontology and mission phase, i.e. something a human analyst would actually write in a log. | "Vehicle detected and tracked, 14:20–14:29, during ISR loiter phase" | Fusion + ontology mapping |

**The ontology itself should be a config file, not hardcoded**, e.g.:

```yaml
mission_phases: [takeoff, transit, loiter/observation, engagement_area, rtb, landing]
event_types:
  - id: object_appearance
    trigger: new_track_id
    min_persistence_s: 1.5
  - id: object_persistence
    trigger: track_duration
    min_persistence_s: 5.0
  - id: object_disappearance
    trigger: track_ended
  - id: count_change
    trigger: class_count_delta
    min_delta: 1
  - id: scene_transition
    trigger: ssim_drop OR camera_motion_spike
  - id: speech_event
    trigger: vad_segment_with_transcript
  - id: mission_phase_change
    trigger: manual_or_heuristic
```

This gives you three things a hardcoded pipeline doesn't: (a) your guide can tune it without touching code, (b) you can defend "why is this an event" in the viva with a rule, not a guess, (c) it's the natural place to add domain-specific classes later (vehicle types, structures) without redesigning the engine.

**Detection vs. operationally-significant event** — the distinguishing signal is *persistence + context*, not confidence. A single-frame flicker at 0.9 confidence is noise; a track that survives 15 frames of your tracker at 0.5 confidence and coincides with a scene-relevant phase is a real event. This is exactly why tracking (not per-frame detection) has to sit upstream of your scoring engine — see §2.

---

## 2. YOLO Strategy — Answering Your 13 Sub-Questions

| # | Question | Recommendation |
|---|---|---|
| 1 | Which YOLO family | **YOLOv8n or YOLO11n** (Ultralytics, AGPL/commercial-clear enough for academic use). Nano variant — anything bigger is wasted on a 4 GB card once you also need VRAM headroom for Whisper/Qwen in the same session. |
| 2 | Detection or tracking | **Tracking.** Raw per-frame detection gives you Tier-0 observations only; you need track IDs for persistence, which is your main event signal. |
| 3 | ByteTrack vs BoT-SORT | **ByteTrack.** It's built into Ultralytics (`model.track(tracker="bytetrack.yaml")`), CPU-cheap, and doesn't need a re-ID embedding model (BoT-SORT does, and that's extra GPU load you don't have room for). |
| 4 | Maintaining IDs across frames | ByteTrack handles short occlusions via IOU + Kalman-filter motion prediction. Set `track_buffer` (frames to keep a lost track alive) to something like 30–60 depending on your sampling rate — UAV footage has more occlusion/re-appearance than static CCTV. |
| 5 | Detections → temporal events | Aggregate per-track: `first_seen`, `last_seen`, `duration`, `class`, `mean_confidence`, `bbox_trajectory`. A track becomes a Tier-1 candidate once `duration ≥ min_persistence_s` from your ontology config. |
| 6 | Confidence handling | Use **per-track mean/median confidence**, not per-frame. A track with fluctuating 0.35–0.6 confidence over 5 seconds is more trustworthy than one 0.9-confidence single-frame blip. |
| 7 | False-positive filtering | (a) persistence threshold, (b) minimum bbox size (filters distant noise), (c) class-conditional confidence floors, (d) camera-motion compensation before tracking (see §6 — this is the single biggest FP source in UAV footage). |
| 8 | Persistence → importance | Feed `duration`, `distance_traveled`, `count_change` directly into the event score (§5) — don't gate on persistence alone or you'll miss legitimate short but critical events (e.g., a brief weapon-flash detection). Use persistence as one *weighted* signal, not a hard filter, except for the noise floor. |
| 9 | Custom training needed? | **Only if pretrained COCO classes miss your domain objects.** COCO has `car`, `truck`, `person`, `boat`, `airplane` — often enough for a first version. If your guide names UAV-specific classes (specific vehicle types, structures, weapons) that aren't in COCO, you need fine-tuning — see below. |
| 10 | Objects not in pretrained model | Three fallback options, cheapest first: (a) map close COCO classes as proxies and label the gap in your report, (b) few-shot fine-tune YOLOv8n on a small labeled set (100–300 images is enough to meaningfully improve a nano model on a narrow class), (c) skip class-specific detection for that object and fall back to generic "unclassified object" + motion/appearance-based event. Do **not** attempt training from scratch — see §14. |
| 11 | Every frame or sampled | **Sampled.** At 1920×1080/30fps, running detection every frame for a 30-minute mission is ~54,000 frames — infeasible on a P2000 in reasonable time. Sample at **2–5 fps** for detection/tracking; this is still dense enough for ByteTrack to maintain continuity. |
| 12 | Inference resolution | Run YOLO inference at **640×640** (standard YOLO input), letterboxed from the 1920×1080 source. Map detected bboxes back to full-res coordinates for the evidence package and for drawing on the (still full-quality) highlight reel. Original footage is never downscaled for the output video — only the *inference copy* is. |
| 13 | Compute scaling with duration | Roughly linear in sampled-frame count. A 4 GB P2000 running YOLOv8n at 640px should do somewhere in the range of 15–40 fps depending on driver/CUDA version — budget conservatively and benchmark in Week 3, don't assume. At 3 fps sampling, a 30-minute video is ~5,400 inference frames, which should process well under real-time. |

**Should YOLO be the primary detector or one signal among several?** **One signal among several.** YOLO tells you *what* and *where*; it says nothing about scene change, camera motion, or audio. Treat it as the most informative single input to the scoring engine (§5), not the decision-maker.

---

## 3. Computer-Vision Pipeline — Handling UAV-Specific Noise

UAV footage has a failure mode ground-camera CV pipelines don't: **the camera itself moves constantly**, so naive frame-diff/SSIM/optical-flow will scream "change!" on every pan, and vibration/turbulence adds high-frequency jitter that looks like motion everywhere.

**Feasible first-version pipeline (10-week budget):**

1. **Camera motion compensation** — before any diff/SSIM/optical-flow step, estimate global camera motion between consecutive sampled frames using sparse optical flow (Lucas-Kanade on corner features, `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`) or a lightweight homography via `cv2.findHomography` on matched ORB features. Warp frame *t* into frame *t-1*'s reference frame, **then** diff.
2. **Frame difference / SSIM on the motion-compensated pair** — now a genuine scene change (new object, lighting change) stands out against a near-static background instead of drowning in camera-pan noise.
3. **YOLO + ByteTrack** running in parallel on the sampled frames (independent of the diff pipeline — object tracking is inherently somewhat motion-invariant because you're tracking the object's own bbox trajectory, not raw pixels).
4. **Fuse the two** — a "visual change" signal from step 2 plus "object event" signal from step 3, both feeding the scorer.

**What to skip in v1:** full homography-based mosaicking, dense optical flow (Farneback is CPU-expensive at 1080p), and anything requiring camera intrinsics/gimbal telemetry you likely don't have. Sparse-feature motion compensation is the right complexity level for 10 weeks.

**Turbulence/vibration:** apply a small temporal smoothing window (e.g., median over 3–5 sampled frames) to the change score before thresholding, so single-frame jitter spikes don't fire events.

---

## 4. Highlight Reel Generation — Adaptive Duration Algorithm

**Pipeline:**

```
events.json (Tier-2, scored, timestamped)
        │
        ▼
1. Sort events chronologically
2. For each event: window = [event.start - pre_context, event.end + post_context]
      pre_context, post_context scale with event.score (higher score → more context,
      bounded by min/max, e.g. 3–10s pre, 3–15s post)
3. Merge overlapping/adjacent windows (gap < merge_threshold, e.g. 2s) into clip groups
4. Enforce min_clip_duration (e.g. 4s — don't emit sub-second unwatchable clips)
      and max_clip_duration (e.g. 45s — split very long persistent-object windows,
      keep start+end, summarize middle in the evidence JSON instead of the video)
5. Redundancy suppression: if two clips show the same track_id with no new event type,
   keep the first and last occurrence only, drop the visually-repetitive middle
6. Always prepend a mission-start clip (first N seconds post-takeoff-detection)
   and append a mission-end clip (last N seconds pre-landing) regardless of score
7. Concatenate remaining clips in chronological order
8. Adaptive total duration = sum of surviving clip durations (no fixed target —
   this is what makes an event-dense 30-min mission produce an 8-min highlight
   and a quiet one produce 4 minutes, exactly as you specified)
```

**FFmpeg implementation** — two viable approaches:

- **Simple/robust (recommended for weeks 5–6):** use `ffmpeg -ss <start> -to <end> -c copy` per clip (stream copy = no re-encode, keeps original 1920×1080 quality, very fast), write clip list, concatenate with the `concat` demuxer:
```bash
ffmpeg -ss 00:14:12 -to 00:14:32 -i mission.mp4 -c copy clip_003.mp4
# repeat per clip, then:
ffmpeg -f concat -safe 0 -i clips.txt -c copy highlight.mp4
```
  Caveat: `-c copy` cuts on keyframe boundaries, so clip start times can drift by up to one GOP. Acceptable for a first version; mention it as a known limitation in your report.
- **Frame-accurate (stretch goal, only if time allows):** re-encode with `-c:v libx264 -crf 18` for exact cuts — costs CPU time and a full re-encode pass, don't attempt this until the simple version works end-to-end.

---

## 5. Event Scoring Framework

```
EventScore = w1·yolo_relevance + w2·persistence_norm + w3·visual_change_norm
           + w4·audio_relevance + w5·mission_phase_weight + w6·count_change_norm
```

- **Normalization:** min-max or z-score each raw signal to [0,1] *within a mission*, not globally — absolute pixel-diff values, for instance, are meaningless without knowing that mission's baseline noise floor.
- **Weights:** **manually configured initially** (this is the honest, defensible answer for a 10-week project — you do not have the labeled data volume to learn weights reliably, and hand-tuned + documented weights are something you can actually justify in a viva). Expose them in a YAML config so your guide can adjust priorities without touching code. Learning weights from data is a legitimate "Future Work" line in your report, not a Week-6 deliverable.
- **Thresholds:** start with a fixed score threshold (e.g., ≥0.4 → candidate event), then optionally make it adaptive per-mission using a percentile of that mission's score distribution (e.g., top 30% of candidate windows), which naturally scales to how "busy" a given mission is.
- **Temporal smoothing:** apply a short moving-average over the raw per-sampled-frame score before thresholding, to avoid single-frame score spikes creating spurious 1-frame events.
- **Event merging:** as in §4 — adjacent/overlapping high-score windows become one event with a combined score (e.g., max, not sum, to avoid double-counting the same physical event detected by two signals).
- **False-positive suppression:** persistence floor + minimum bbox area + camera-motion-compensated visual change (§3) together handle most of it; don't try to eliminate FPs purely by raising the score threshold — that just trades false positives for false negatives.

---

## 6. Mission Evidence Package — Is This the Right Architecture?

**Yes — this is the correct approach, and it's the single best architectural decision in your proposal.** No current locally-deployable model, VLM included, can ingest a raw 30-minute 1080p MP4 directly and reason over it coherently; even cloud VLMs handle video by internally sampling frames. Structuring the evidence explicitly (rather than hoping the model "watches" the video) is exactly right, and it's also what makes the system's outputs auditable — a DRDO reviewer can trace every claim in the summary back to a specific keyframe/timestamp/transcript line, which matters a lot more in this domain than in a consumer app.

Your directory layout is sound. One addition: keep a `provenance` field in every JSON linking back to the exact source (frame number, track ID, transcript segment ID) so the LLM prompt (§9) can be told to cite evidence, and so you can compute grounding/hallucination metrics (§12) later.

**Pipeline:**

```
raw video ──▶ media inspector (ffprobe: has audio? resolution? duration? codec?)
     │
     ├──▶ VIDEO PATH: sampled-frame extraction ──▶ YOLO+ByteTrack ──▶ CV change
     │         detection ──▶ event scoring (§5) ──▶ events.json + tracks.json
     │         ──▶ keyframe selection (1-3 representative frames per Tier-2 event,
     │             picked as the frame closest to peak score / peak bbox size)
     │
     ├──▶ AUDIO PATH (if audio exists): ffmpeg extract wav ──▶ VAD ──▶
     │         Faster-Whisper ──▶ transcript.json (§8)
     │
     └──▶ FUSION: merge events.json + transcript.json on the shared timeline
               ──▶ evidence.json (chronological, single source of truth)
                    ──▶ LLM prompt (§9) ──▶ summary.json/txt ──▶ TTS + PDF
```

This matches your proposed diagram closely — the one structural change I'd make is explicit **keyframe selection as its own step** (not "just dump all frames"), because keyframe count directly drives your VLM cost, which matters a lot on a 4 GB card.

---

## 7. The Qwen Question — This Is the Part to Get Right

**Difference between Qwen3 and Qwen3-VL:** Qwen3 is text-only — it can reason superbly over structured JSON/text evidence but cannot look at an image. Qwen3-VL adds a vision encoder and can accept images (your keyframes) directly alongside text, at the cost of significantly more VRAM and slower inference per call.

**Recommendation for your hardware (P2000, 4 GB): a hybrid, sequential architecture — not "VLM does everything."**

1. **Primary reasoning: text-only Qwen3, quantized (Q4_K_M GGUF, ~4B–8B parameter class), via Ollama.** This is the model that reads `evidence.json` (fused events + transcript + mission metadata) and writes the operational summary. Text-only inference at this quantization comfortably fits a 4 GB card and is fast.
2. **Vision, used sparingly: a small VL model (Qwen2.5-VL-3B-Instruct or the smallest available Qwen3-VL variant, quantized) captions only the top-N keyframes** (e.g., 10–20 per mission, selected as in §6) — not every frame, not the whole video. Each caption becomes a short text field attached to its event in `evidence.json`. This step runs **before** the text model, sequentially, so the two never compete for VRAM at the same time.
3. Result: Qwen3 (text) never needs to "see" pixels — it reasons over event metadata + track data + transcript + the VL model's keyframe captions, all as text. This is realistic on your hardware and still gives you genuine multimodal grounding, because the visual information reached the model — just via a caption bridge instead of raw pixels at reasoning time.

**Why not send keyframes straight into Qwen3-VL for the whole summary in one call, as originally proposed?** On 4 GB, a VL model holding several keyframes in context at once for a single long-reasoning call is the riskiest OOM point in the whole system, and it's also the hardest failure to recover from mid-pipeline. Splitting "look at the image" (many small, cheap, independent VL calls) from "reason over everything" (one text-only call with a bigger context window) is both more memory-safe and, honestly, plays to what each model type is actually good at.

**Context maintenance:** each keyframe caption should carry its `event_id` and `timestamp` so the text model's evidence JSON reads as one coherent chronological document (§8), not a disconnected image-caption dump.

**If your guide specifically wants a "true" end-to-end VLM demo** for the viva, keep it as an **optional stretch component**: a single Qwen-VL call on 3–5 hand-picked keyframes for a live demo, clearly labeled as a supplementary capability, not the load-bearing path of the pipeline.

---

## 8. Context Preservation — Windowing Design

For a detected event at *t*, don't extract a fixed ±2s window. Build a **context chain**:

```
pre_context:  walk backward from event.start while the same track_id (or a spatially
              correlated track) is active or a related scene-change is nearby,
              up to a max pre_context cap (e.g. 15s)
core_event:   event.start → event.end as scored
post_context: walk forward the same way, up to a max post_context cap
```

Represent this in **two places**:
- **In the highlight video** — the merged clip window from §4 already encodes this.
- **In the evidence JSON given to the LLM** — as an explicit `context` object per event:
```json
{
  "event_id": "evt_014",
  "type": "vehicle_detected",
  "core": {"start": "14:22:03", "end": "14:22:09"},
  "pre_context_summary": "UAV approaching area from 14:12; no prior detections",
  "post_context_summary": "Vehicle tracked continuously until leaving frame at 14:29:41",
  "related_events": ["evt_012", "evt_015"]
}
```
This is what lets Qwen3 write "the vehicle was tracked for approximately 7 minutes before leaving the frame" instead of a series of disconnected single-timestamp bullet points.

---

## 9. Audio Pipeline & Multilingual Strategy

```
ffmpeg (extract mono 16kHz wav) ──▶ VAD (webrtcvad or Silero-VAD, cheap, filters silence)
   ──▶ Faster-Whisper (CTranslate2, int8 or int8_float16 quantization)
        - model size: "small" or "medium" — "large-v3" is likely overkill for a
          P2000 shared with YOLO/Qwen in the same pipeline run; benchmark both
        - language: auto-detect per segment (Whisper does this natively)
   ──▶ transcript.json: per-segment {start, end, text, language, confidence}
```

- **Mixed-language (Hindi/English code-switching):** Whisper handles this reasonably at segment granularity (it'll tag each segment with its detected dominant language) but will not do word-level language switching within a segment reliably — document this as a known limitation, don't oversell it.
- **Word-level timestamps:** enable them (`word_timestamps=True` in faster-whisper) — cheap to compute and useful for precisely aligning "operator says X" with the visual event it refers to.
- **Radio/noisy audio:** VAD + a `medium` (not `small`) model materially helps here; also normalize audio gain before transcription.
- **Translate before giving to Qwen, or keep original?** **Keep the original-language transcript as the source of truth, and generate an English-normalized version alongside it, not instead of it.** Feed Qwen3 both: the original (for fidelity/audit) and the English version (for reasoning, since your prompt and most of your documentation will be in English). This avoids losing meaning to a lossy auto-translate step while still giving the model something it can reason over consistently.

---

## 10. TTS / Multilingual Briefing

**Realistic for 10 weeks: Option C, but scoped down** — support both original-language and English TTS, but don't chase "every language perfectly." Concretely:

- **Offline TTS engine: Piper** (lightweight, CPU-fine, no GPU needed — frees your P2000 for the stages that actually need it) or **Coqui TTS** if you need broader multilingual voice coverage and can spare more compute. Piper is the safer pick given your hardware budget, since it runs comfortably on CPU while YOLO/Whisper/Qwen occupy the GPU elsewhere in the pipeline.
- **Language → voice selection:** map the transcript's detected dominant language to an available offline voice model; if no voice exists for a detected language, **fall back to the English-normalized summary** rather than failing the briefing step entirely. Document this fallback explicitly — it's a legitimate engineering decision, not a gap to hide.
- Treat "flawless native-quality multilingual TTS for arbitrary Indian languages" as **out of scope** — pretrained offline voices are realistically available for a handful of languages (English, Hindi being the two you most likely need); anything beyond that is a stretch goal.

---

## 11. Multimodal Fusion — The Chronological Evidence Representation

Merge `events.json` (visual) and `transcript.json` (audio) on a single sorted timeline, keyed by timestamp, with mission metadata (phase, takeoff/landing markers) as top-level context. This is exactly the representation you sketched in your prompt (the `00:03 TAKEOFF / 04:11 Vehicle detected / 04:14 Operator says...` example) — that's the right shape. Store it as `evidence.json`; it is the **single object** that gets serialized into the LLM prompt, nothing else needs to be passed ad hoc.

```json
{
  "mission_id": "mission_001",
  "duration_s": 1830,
  "mission_phases": [
    {"phase": "takeoff", "start": "00:00:00", "end": "00:00:45"},
    {"phase": "loiter", "start": "00:00:45", "end": "00:28:00"},
    {"phase": "landing", "start": "00:28:00", "end": "00:30:30"}
  ],
  "timeline": [
    {"t": "00:00:03", "type": "mission_event", "subtype": "takeoff", "confidence": 0.95},
    {"t": "00:04:11", "type": "visual_event", "subtype": "vehicle_detected",
     "track_id": 17, "class": "car", "confidence": 0.81, "event_id": "evt_014",
     "keyframe_caption": "White sedan on unpaved road, stationary near tree line"},
    {"t": "00:04:14", "type": "speech_event", "language": "en", "confidence": 0.88,
     "text_original": "Vehicle spotted ahead.", "text_en": "Vehicle spotted ahead."},
    {"t": "00:04:25", "type": "visual_event", "subtype": "object_left_frame",
     "track_id": 17, "event_id": "evt_014"}
  ]
}
```

---

## 12. LLM Prompting & JSON Schemas

**Prompt design principles:** ground strictly in supplied evidence, forbid inference beyond it, preserve chronology, separate "detected" (machine-observed) from "confirmed" (operator-stated) language, flag uncertainty explicitly, cite `event_id`s.

**Conceptual system prompt:**
```
You are a mission-analysis assistant. You will receive a JSON evidence timeline
from a UAV mission (evidence.json). You must:
1. Use ONLY the information present in the JSON. Do not infer objects, actions,
   or outcomes that are not explicitly recorded.
2. Preserve strict chronological order in your narrative.
3. Distinguish machine detections ("a vehicle was detected") from operator
   statements ("the operator identified the vehicle as...") — never merge them
   into one claim.
4. For every claim, reference the event_id(s) it is based on.
5. If evidence is sparse, low-confidence, or ambiguous, state that explicitly
   rather than resolving the ambiguity yourself.
6. Output valid JSON matching the provided summary.json schema.
```

**Schemas (trimmed to essentials — expand as needed):**

`events.json` — array of `{event_id, type, subtype, start, end, track_ids[], score, confidence}`
`transcript.json` — array of `{segment_id, start, end, language, text_original, text_en, confidence}`
`evidence.json` — as shown in §11 (fused timeline + phases)
`summary.json`:
```json
{
  "mission_id": "mission_001",
  "narrative": "string, chronological, cites event_ids",
  "key_events": [{"event_id": "evt_014", "description": "...", "confidence": "high|medium|low"}],
  "uncertainties": ["string list of anything the model flagged as ambiguous"],
  "mission_outcome": "string, only if directly supported by evidence"
}
```

---

## 13. Final Recommended System Architecture

```
UAV VIDEO (1920x1080)
        │
        ▼
  ffprobe (media inspector: has audio? duration? codec?)
        │
   ┌────┴─────────────────────────┐
   ▼                               ▼
VIDEO PATH                    AUDIO PATH (if present)
sampled frames (2-5fps)       ffmpeg extract wav
   │                               │
   ├─▶ YOLO11n + ByteTrack         ├─▶ VAD
   ├─▶ motion-compensated          ├─▶ Faster-Whisper (int8)
   │   frame diff/SSIM             └─▶ transcript.json
   │
   └─▶ Event Scoring Engine (§5) ─▶ events.json, tracks.json
                │
                ├─▶ Keyframe selection (top-N per event)
                │       └─▶ small VL model (Qwen2.5-VL / Qwen3-VL small, quantized)
                │              → keyframe_captions
                │
                └─▶ Highlight Generator (§4, FFmpeg) ─▶ highlight.mp4 (full 1080p)

Event fusion: events.json + transcript.json + keyframe_captions + mission metadata
        │
        ▼
  evidence.json  ◀── this is what's saved permanently as the "record of evidence"
        │
        ▼
  Qwen3 (text, quantized, via Ollama)  ─▶ summary.json / summary.txt
        │
        ├─▶ TTS (Piper) ──▶ briefing.wav
        └─▶ ReportLab ──▶ report.pdf

        ▼
  Streamlit UI — surfaces highlight.mp4, summary, briefing audio, PDF, evidence.json
```

Key change from your original diagram: **Qwen3-VL is not a single monolithic stage** — it's split into a cheap keyframe-captioning sub-stage (feeding evidence.json) and the main reasoning stage runs as text-only Qwen3. This is the change that makes the architecture survive on 4 GB VRAM.

---

## 14. Offline / Air-Gapped Deployment

**Provisioning strategy — build the wheelhouse and model cache at home, transfer once, install offline at DRDO.**

1. **Pin your target environment first**, before downloading anything:
   - Python version (recommend 3.10 or 3.11 — best current wheel availability for torch/onnxruntime as of major package ecosystems)
   - OS: Windows, get exact version from the facility PC (`winver`)
   - Architecture: `win_amd64` almost certainly, confirm it's not ARM64
   - GPU: Quadro P2000 → confirm installed driver version and CUDA capability *on that specific machine* (`nvidia-smi` when you're physically there Thu/Fri) before choosing a CUDA-enabled torch build; if driver/CUDA version is uncertain, default to CPU-only wheels for torch as the safe fallback and treat GPU acceleration as an opportunistic upgrade once confirmed.

2. **Wheel compatibility mechanics:**
   - Wheel filename encodes everything: `torch-2.x.x-cp311-cp311-win_amd64.whl` → `cp311` = CPython 3.11 built extension, `win_amd64` = platform tag. Both must match the target machine exactly.
   - Build the wheelhouse with `pip download -r requirements.txt -d wheelhouse/ --python-version 3.11 --platform win_amd64 --only-binary=:all:` — this resolves and downloads the **full transitive dependency tree**, not just top-level packages.
   - Record hashes: `pip download ... --no-binary :none:` plus a `pip freeze > versions.txt` / `pip hash` record for reproducibility and integrity verification after transfer.

3. **Minimize wheelhouse + model size** (this matters a lot given your Drive/email-only transfer constraint):
   - Use **CPU-only PyTorch wheels** unless you've confirmed the CUDA build works on the P2000's actual driver — CPU wheels are dramatically smaller and avoid a whole class of CUDA-version-mismatch failures.
   - Use **quantized GGUF models for Ollama** (Q4_K_M class) — a 4B-class Qwen3 GGUF at Q4 is on the order of 2-3 GB rather than the multi-GB fp16 original.
   - Use `faster-whisper`'s **small/medium int8 CTranslate2 models** (hundreds of MB, not GB).
   - Use **YOLO nano** weights (~6 MB).
   - Use **Piper voices** (tens of MB each).
   - **Total realistic package: low single-digit GB**, which is transferable via Drive across a few sessions even without USB — split into multiple zip parts if needed to stay under any per-file size limits the facility's network imposes.

4. **Offline wheelhouse structure:**
```
offline_package/
  wheelhouse/           # all .whl files, full dependency tree
  models/
    yolo/yolo11n.pt
    whisper/faster-whisper-small-int8/
    ollama/qwen3-4b-q4.gguf
    ollama/qwen-vl-3b-q4.gguf   (optional keyframe-captioning model)
    piper/en_voice.onnx
    piper/hi_voice.onnx
  requirements.txt
  versions.txt
  hashes.sha256
  install.bat
```

5. **Verify integrity after transfer** — checksum every file against `hashes.sha256` before installing; a corrupted transfer via Drive/email is a realistic failure mode you should explicitly test for, not discover during a demo.

---

## 15. `install.bat` — Design

Structure it as numbered, individually-loggable steps, each writing to `install_log.txt`, and **fail loudly with a clear message rather than silently continuing** — an air-gapped machine has no StackOverflow to fall back on if something goes wrong mid-install.

```batch
@echo off
setlocal enabledelayedexpansion
set LOG=install_log.txt
echo Install started: %date% %time% > %LOG%

:: 1. Check Python version
python --version >> %LOG% 2>&1 || (echo Python not found & goto :fail)

:: 2. Check architecture
python -c "import platform; print(platform.machine())" >> %LOG%

:: 3. Create venv
python -m venv venv >> %LOG% 2>&1 || goto :fail

:: 4. Install from local wheelhouse only (--no-index forces offline)
venv\Scripts\pip install --no-index --find-links=wheelhouse -r requirements.txt >> %LOG% 2>&1 || goto :fail

:: 5. Verify critical imports
venv\Scripts\python -c "import cv2, torch, ultralytics, faster_whisper, streamlit, reportlab" >> %LOG% 2>&1 || goto :fail

:: 6. Check ffmpeg
ffmpeg -version >> %LOG% 2>&1 || (echo FFmpeg missing from PATH & goto :fail)

:: 7-10. Verify YOLO weights, Whisper model, Ollama binary + model pulled locally
:: (each as its own check, logged, non-fatal warnings vs fatal errors distinguished)

:: 11. Verify Piper TTS voice files present

:: 12. Verify Streamlit launches (headless smoke test)
venv\Scripts\streamlit hello --server.headless true >> %LOG% 2>&1

:: 13. Smoke test: run pipeline on a 10-second sample clip end-to-end

echo Install completed successfully: %date% %time% >> %LOG%
goto :eof

:fail
echo INSTALL FAILED - see %LOG% for details
exit /b 1
```

---

## 16. Git Workflow & Phase Structure

**Branching:** `main` (always working), `dev` (integration), `phase/N-name` feature branches merged into `dev` then `main` at each Wednesday release-candidate point.

**Commit convention:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`) — small enough overhead to be worth it for a guide-reviewed project.

**Tags:** `v0.1-phase1` ... `v1.0-final`, tagged every Wednesday RC.

**What NOT to commit:** raw UAV footage (sensitive + large — `.gitignore` the `input/` and `output/` mission folders), model weights and GGUF files (large — document the wheelhouse/model provisioning process instead, don't version large binaries in Git), any facility-specific credentials/paths, `venv/`.

**Phase structure (your proposal is good, minor reorder to match the hybrid Qwen decision):**

| Phase | Focus |
|---|---|
| 1 | Environment + offline wheelhouse pipeline |
| 2 | Media ingestion (ffprobe, frame sampling) |
| 3 | YOLO + ByteTrack |
| 4 | CV change detection + event scoring engine |
| 5 | Highlight generation (FFmpeg) |
| 6 | Faster-Whisper audio pipeline |
| 7 | Evidence fusion (events + transcript → evidence.json) + keyframe captioning |
| 8 | Qwen3 (text) reasoning + prompt/schema iteration |
| 9 | TTS + PDF report + Streamlit UI |
| 10 | Offline install validation + end-to-end testing at DRDO |

---

## 17. 10-Week Plan (Mon/Tue dev, Wed RC, Thu/Fri DRDO)

| Wk | Objective | Mon–Tue | Wed | Thu–Fri (DRDO) | Deliverable / Tag | Acceptance | Risk |
|---|---|---|---|---|---|---|---|
| 1 | Env + wheelhouse | Pin Python/OS target, build initial wheelhouse for core libs (opencv, torch-cpu, ultralytics) | Package + integrity-check wheelhouse | Confirm exact facility spec (`nvidia-smi`, `winver`), clarify approved transfer channel with guide | `v0.1-phase1` | `install.bat` runs clean on a test VM matching facility spec | Transfer-channel policy unresolved |
| 2 | Media ingestion | ffprobe wrapper, frame sampler, project skeleton, Git setup | RC: ingestion module | Test on real sample UAV footage at DRDO if available | `v0.2-phase2` | Correctly reports audio presence/absence, resolution, duration on 3+ test clips | No representative sample footage |
| 3 | YOLO + tracking | Integrate YOLO11n + ByteTrack, benchmark fps on target-spec hardware | RC: tracking module + benchmark numbers | Run on P2000 directly, record real fps/VRAM | `v0.3-phase3` | Stable track IDs across a 2-min test clip | P2000 slower than expected — fallback to lower sampling fps |
| 4 | Event scoring | Motion-compensated diff, event ontology config, scoring engine | RC: events.json output | Guide review of ontology + weights | `v0.4-phase4` | Precision/recall sanity-checked against 1 manually-labeled clip | Weight tuning takes longer than a day |
| 5 | Highlight gen | FFmpeg clip extraction/merge, adaptive duration logic | RC: highlight.mp4 pipeline | Validate output quality/timing on real footage | `v0.5-phase5` | Highlight preserves takeoff/landing + top events, chronological | `-c copy` keyframe-boundary drift |
| 6 | Audio pipeline | Faster-Whisper integration, VAD, multilingual test | RC: transcript.json | Test with real/recorded operator audio if available | `v0.6-phase6` | Transcript + timestamps on English + Hindi test clips | Multilingual accuracy weaker than hoped — document limitation |
| 7 | Fusion + captioning | evidence.json fusion, keyframe selection, integrate small VL model | RC: evidence.json | Confirm VL model fits VRAM budget on actual P2000 | `v0.7-phase7` | evidence.json chronologically correct, captions attached | VL model OOM — fallback: skip captions, text-only evidence |
| 8 | Qwen3 reasoning | Ollama + Qwen3 GGUF integration, prompt engineering, JSON schema enforcement | RC: summary.json | Run on real evidence.json from DRDO test footage | `v0.8-phase8` | Summary grounded, cites event_ids, no fabricated events on test set | Hallucination — tighten prompt, add stricter schema validation |
| 9 | TTS + PDF + UI | Piper TTS, ReportLab report, Streamlit dashboard | RC: full UI wired to pipeline | Guide walkthrough of full UI | `v0.9-phase9` | End-to-end: upload video → all outputs generated in Streamlit | UI polish time pressure — keep it functional, not fancy |
| 10 | Deployment validation | Fix bugs from Wk9 review, finalize docs | RC: final build | Full offline install + demo run on air-gapped PC, timing/resource logging | `v1.0-final` | Clean install from wheelhouse only, full pipeline run end-to-end offline | Any late-discovered offline-only dependency gap |

---

## 18. Testing & Validation Framework

| Area | Metric | How to measure |
|---|---|---|
| Event detection | Precision/Recall/F1 | Manually label events on 3-5 clips as ground truth; compare against engine output with IoU-style temporal overlap matching |
| Highlight generation | Event coverage %, compression ratio | (events preserved in highlight / total scored events); (highlight duration / source duration) |
| Highlight quality | Redundancy, contextual completeness | Human rubric scoring (1-5) from guide/peers on a sample set |
| ASR | WER on a small manually-transcribed sample, language-ID accuracy | Compare Whisper output to hand transcript; confusion matrix for language ID |
| LLM | Grounding rate, hallucination rate | For each claim in summary.json, manually check it traces to a real event_id in evidence.json; hallucination = claim with no supporting event_id |
| System | Wall-clock time, peak VRAM/RAM, CPU% | `nvidia-smi` logging during pipeline run, Python `resource`/`psutil` instrumentation, tested on the actual P2000 target |

---

## 19. Risks

| Risk | Prob. | Impact | Mitigation | Fallback |
|---|---|---|---|---|
| YOLO false positives from camera motion | High | Med | Motion compensation (§3), persistence filtering | Manual score-threshold tuning per mission |
| Domain objects missing from pretrained YOLO | Med | Med | COCO proxy classes; small fine-tune set if time allows | Report class-coverage gap explicitly, don't fake it |
| P2000 VRAM insufficient for VL+text models together | High if run concurrently | High | Strictly sequential pipeline (§7), CPU-only fallback for TTS/text stages | Drop keyframe captioning, text-only evidence to Qwen3 |
| Offline transfer channel (Drive/email) blocked or too slow | Med-High | High | Minimize package size (§14), test transfer in Week 1, escalate to guide immediately if blocked | Request approved alternative transfer method from facility IT |
| Whisper struggles with mixed Hindi/English | Med | Med | Segment-level language tagging, document limitation | English-only transcript mode as fallback |
| Qwen hallucination in summaries | Med | High (credibility) | Strict grounding prompt, JSON schema validation, cite event_ids | Human-in-the-loop review flag on low-confidence summaries |
| Long videos → long processing time | Med | Med | Frame sampling (not every frame), benchmark early | Cap max processed duration, chunked processing |
| Incompatible wheel/CUDA version discovered late | Med | High | Confirm exact driver/CUDA on P2000 in Week 1 visit | CPU-only builds throughout as safe default |
| Insufficient storage on facility PC for models+footage | Low-Med | Med | Confirm available disk in Week 1 | Aggressive model quantization, prune old mission data |

---

## 20. MUST / SHOULD / NICE / DO NOT

**MUST HAVE**
- Media inspection (audio presence, resolution, duration)
- YOLO + ByteTrack event detection with camera-motion-compensated CV change signal
- Adaptive-length highlight generation preserving takeoff/landing
- Faster-Whisper transcription with language detection
- Text-only Qwen3 (Ollama) generating a grounded summary from structured evidence
- PDF report + Streamlit UI
- Fully offline install path with a documented, tested wheelhouse

**SHOULD HAVE**
- Keyframe captioning via a small quantized VL model (sequential, not concurrent with the text model)
- TTS briefing (Piper) in at least English + one additional language
- Event ontology as an editable config file
- Basic precision/recall evaluation against a hand-labeled test set

**NICE TO HAVE**
- Fine-tuned YOLO on a small custom class set
- Frame-accurate (re-encoded) highlight cuts instead of stream-copy cuts
- Word-level ASR timestamp display in the UI
- Adaptive/learned event-scoring weights

**DO NOT ATTEMPT**
- Real-time/live-stream processing — this is an offline batch pipeline, full stop
- Every-frame VLM inference over the whole video
- Training YOLO from scratch (fine-tuning a pretrained model is the ceiling here)
- Supporting more than 2-3 languages end-to-end
- Multi-camera / sensor-fusion beyond single UAV video+audio
- A large (13B+) LLM — won't fit your VRAM budget and won't run fast enough to be demoable
- Building a database/backend service layer — flat files (JSON) are sufficient and easier to defend as "auditable evidence" in a DRDO context anyway
- Cloud integration of any kind, even "just for testing" — build offline-first from day one so you're not retrofitting it in Week 9

---

## 21. Final Recommendation Summary

1. **Project definition:** an offline, evidence-grounded UAV mission analysis pipeline that turns raw footage into an adaptive highlight reel, a multilingual transcript, an auditable structured evidence record, and an LLM-generated operational summary with audio briefing and PDF report — all running on a single air-gapped Windows workstation with a 4 GB GPU.
2. **Architecture:** as in §13 — the key departure from your draft is splitting vision (cheap, sparse keyframe captioning) from reasoning (text-only Qwen3 over structured evidence), run strictly sequentially.
3. **Stack:** Python 3.11, OpenCV, Ultralytics YOLO11n, ByteTrack, Faster-Whisper (CTranslate2, int8), Ollama + quantized Qwen3 (text) + optional small quantized VL model, Piper TTS, ReportLab, Streamlit, FFmpeg.
4. **YOLO:** nano model, 2-5 fps sampling, 640px inference resolution mapped back to full-res, ByteTrack, COCO classes with documented gaps, fine-tune only if a labeled set materializes.
5. **Whisper:** small/medium, int8, VAD-gated, auto language-ID per segment, original + English-normalized transcript both preserved.
6. **Qwen strategy:** text-only Qwen3 for reasoning; small quantized VL model only for sparse keyframe captioning, never both loaded concurrently.
7. **Ollama:** local model pull done once (at home, with internet), GGUF file transferred as part of the offline package, served locally at DRDO with no network calls.
8. **TTS:** Piper, English + one additional language, explicit fallback to English if a detected language has no available voice.
9. **Git:** phase-branch workflow, Wednesday RC tags, no large binaries or raw footage committed.
10. **Offline install:** wheelhouse built and hash-verified at home, transferred via the facility's approved channel (resolve this explicitly with your guide in Week 1), `install.bat` with per-step logging and a smoke test.
11. **Roadmap:** §17, 10 weeks, aligned to your Mon-Wed/Thu-Fri schedule.
12. **Major risks:** VRAM contention on the P2000, the transfer-channel question, camera-motion-induced false positives, LLM hallucination — all have concrete mitigations above.
13. **MVP:** §20 "MUST HAVE" list — that alone is a complete, demoable, defensible system.
14. **Final demonstrable system:** upload a UAV mission video in Streamlit → see media info, an adaptive full-quality highlight reel, a timestamped multilingual transcript, a grounded operational summary citing specific events, an audio briefing, and a PDF report — generated entirely offline on the DRDO workstation.

### Is this realistically achievable in 10 weeks?

**Yes, for the MUST HAVE scope, run on your actual schedule — but only if you resolve the transfer-channel question in Week 1 (not discover it's a blocker in Week 8) and validate your P2000's real throughput/VRAM behavior by Week 3 rather than assuming it from spec sheets.** The SHOULD HAVE items (keyframe captioning, bilingual TTS) are realistic add-ons if the MUST HAVE core lands on schedule by roughly Week 8. Do not let the VLM ambition or the "support many languages" ambition creep back into the MVP path — both are exactly the kind of scope that looks small in a proposal document and consumes two extra weeks in practice.
