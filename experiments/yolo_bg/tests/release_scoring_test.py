#!/usr/bin/env python3
"""Check release stdout event scoring against the historical JSON scorer."""
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

release = load('release_score', ROOT / 'experiments/yolo_bg/summarize_release_batch.py')
legacy = load('legacy_score', ROOT / 'score_falls.py')

# Thirty consecutive callbacks are one event, including when it crosses a GT
# endpoint. An additional event within that GT is a duplicate, not FP or FN.
frames = list(range(20, 50)) + list(range(70, 100)) + list(range(150, 180))
runs = release.segment(frames, 15)
assert runs == [(20, 49), (70, 99), (150, 179)]
assert release.score_segments([(100, 400), (1000, 1100)], runs) == (
    1, 1, 1, 1, [(150, 179)])
assert release.segment([31, 0, 15, 15], 15) == [(0, 15), (31, 31)]
assert release.score_segments([], []) == (0, 0, 0, 0, [])
assert release.score_segments([], [(3, 32)]) == (0, 1, 0, 0, [(3, 32)])
# Preserve the legacy overlap rule, even if one detection spans two GT events.
assert release.score_segments([(20, 30), (50, 60)], [(5, 15)])[:3] == (2, 0, 0)

rng = random.Random(210)
for _ in range(1000):
    frames = [rng.randrange(300) for _ in range(rng.randrange(100))]
    gt = [(start, start + rng.randrange(80)) for start in (rng.randrange(1200) for _ in range(rng.randrange(8)))]
    gap = rng.randrange(1, 30)
    runs = release.segment(frames, gap)
    assert runs == legacy.segment(frames, gap)
    expected = legacy.score_one(gt, runs, 4)
    actual = release.score_segments(gt, runs)
    assert actual[:3] == expected[:3]
    assert actual[4] == expected[4]
print('PASS: callback consolidation, GT endpoint overlap, duplicates, missed/negative cases; 1000 cases match score_falls.py')
