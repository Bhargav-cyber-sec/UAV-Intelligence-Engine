import json
from collections import Counter

with open('data/output/sample_mission/standard/tracks.json') as f:
    tracks = json.load(f)
with open('data/output/sample_mission/standard/detections.json') as f:
    dets = json.load(f)

print('=== TRACK ANALYSIS ===')
print(f'Total tracks: {len(tracks)}')
print(f'Total detections: {len(dets)}')

class_counts = Counter(t['class_name'] for t in tracks)
print(f'Classes detected: {dict(class_counts)}')
print()

print('=== SHORT-LIVED TRACKS (noise, < 3 frames) ===')
short = [t for t in tracks if t['frame_count'] < 3]
print(f'Count: {len(short)} (out of {len(tracks)} total)')

print()
print('=== PERSISTENT TRACKS (>= 5 frames) ===')
persistent = [t for t in tracks if t['frame_count'] >= 5]
for t in persistent:
    print(f"  Track#{t['track_id']} class={t['class_name']} frames={t['frame_count']} mean_conf={t['mean_confidence']:.3f} first={t['first_frame']} last={t['last_frame']}")

print()
print('=== CONFIDENCE DISTRIBUTION ===')
confs = [t['mean_confidence'] for t in tracks]
below_05 = [c for c in confs if c < 0.5]
between_05_07 = [c for c in confs if 0.5 <= c < 0.7]
above_07 = [c for c in confs if c >= 0.7]
print(f'  < 0.50 (weak/noise): {len(below_05)}')
print(f'  0.50-0.70 (ok):      {len(between_05_07)}')
print(f'  >= 0.70 (strong):    {len(above_07)}')

print()
print('=== FRAME COVERAGE ===')
frames_with_dets = len(set(d['frame_index'] for d in dets))
total_frames = 66
print(f'  Frames with detections: {frames_with_dets} / {total_frames}')
print(f'  Frames with NO detections: {total_frames - frames_with_dets}')
