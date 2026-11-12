#!/usr/bin/env python3
"""Score merged fall events, matching score_falls.py (gap=15 gray frames, stride=4)."""
import argparse
import csv
import re
from pathlib import Path


def segment(frames, gap):
    """Merge sorted unique callback frames when the adjacent difference <= gap."""
    runs = []
    for frame in sorted(set(frames)):
        if runs and frame - runs[-1][1] <= gap:
            runs[-1] = (runs[-1][0], frame)
        else:
            runs.append((frame, frame))
    return runs


def score_segments(intervals, runs):
    """Same interval-overlap scoring as legacy score_falls.score_one(..., 4).

    Each GT interval is TP if any merged detection overlaps it. Unmatched GT
    intervals are FN. Only whole merged detections with no GT overlap are FP.
    Extra matched detections are duplicates; they do not add FP or FN.
    """
    matched_gt = set()
    matched_runs = set()
    for i, (start, end) in enumerate(intervals):
        for j, (first, last) in enumerate(runs):
            if first * 4 <= end and last * 4 >= start:
                matched_gt.add(i)
                matched_runs.add(j)
    tp = len(matched_gt)
    fp_runs = [run for j, run in enumerate(runs) if j not in matched_runs]
    return tp, len(fp_runs), len(intervals) - tp, max(0, len(matched_runs) - tp), fp_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    root = Path(__file__).resolve().parents[2]
    default_gt = root / 'groundTruth'
    if not default_gt.is_dir():
        default_gt = root / 'Bug_Video/Fall_17/groundTruth'
    parser.add_argument('--ground-truth', type=Path, default=default_gt)
    parser.add_argument('--output', type=Path, help='Accuracy CSV destination (default: batch/accuracy.csv)')
    parser.add_argument('--gap', type=int, default=15, help='Merge adjacent callback frames if difference <= gap (gray frames; default 15)')
    args = parser.parse_args()
    if args.gap < 1:
        parser.error('--gap must be at least 1')
    with (args.batch / 'status.csv').open() as f:
        statuses = list(csv.DictReader(f))
    if not statuses or any(s['status'] != 'completed' for s in statuses):
        parser.error('Batch is empty or contains failed/incomplete videos')
    settings = dict(line.split('=', 1) for line in (args.batch / 'run.txt').read_text().splitlines())
    if settings.get('max_frames') != '0':
        parser.error('Accuracy requires every frame; MAX_FRAMES must be 0')
    rows = []
    for item in statuses:
        video = item['video']
        text = (args.batch / video / 'runner.log').read_text(errors='replace')
        falls = [int(x) for x in re.findall(r'^\[Callback\] Fall detected at frame (\d+)', text, re.M)]
        counts = re.search(r'^FALL_COUNT=(\d+) BEDEXIT_COUNT=(\d+)', text, re.M)
        if not counts or int(counts[1]) != len(falls):
            parser.error(f'{video}: incomplete or inconsistent event log')
        truth_path = args.ground_truth / f'{video}_fall_frameindex.txt'
        if not truth_path.is_file():
            parser.error(f'Missing ground truth: {truth_path}')
        intervals = []
        for line in truth_path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            start, end = map(int, line.split(','))
            if start == end == 0:  # Dataset sentinel: no falls.
                continue
            if start < 0 or end < start:
                parser.error(f'Invalid ground truth: {truth_path}: {line}')
            intervals.append((start, end))
        runs = segment(falls, args.gap)
        tp, fp, fn, duplicates, fp_runs = score_segments(intervals, runs)
        stable = re.search(r'^BG_STABLE sdk_frame_index=(\d+) gray_index=(\d+) timestamp_ms=(\d+)', text, re.M)
        rows.append(dict(video=video, TP=tp, FP=fp, FN=fn, duplicate_events=duplicates,
                         outside_events=fp, fall_callbacks=len(falls),
                         detected_events=len(runs), repeated_callbacks=len(falls)-len(runs),
                         event_gap_gray=args.gap,
                         detected_segments_gray=';'.join(f'{a}-{b}' for a,b in runs),
                         fp_segments_gray=';'.join(f'{a}-{b}' for a,b in fp_runs), Recall=tp / (tp + fn) if tp + fn else '',
                         Precision=tp / (tp + fp) if tp + fp else '',
                         first_stable_sdk_index=stable[1] if stable else 'NA',
                         first_stable_gray_index=stable[2] if stable else 'NA',
                         first_stable_timestamp_ms=stable[3] if stable else 'NA'))
    output = args.output if args.output is not None else args.batch / 'accuracy.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['video']}: TP={row['TP']} FP={row['FP']} FN={row['FN']} duplicates={row['duplicate_events']} BG_gray={row['first_stable_gray_index']}")
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('TP', 'FP', 'FN'))
    recall = f'{tp / (tp + fn):.2%}' if tp + fn else 'NA'
    precision = f'{tp / (tp + fp):.2%}' if tp + fp else 'NA'
    print(f'TOTAL TP={tp} FP={fp} FN={fn} Recall={recall} Precision={precision}')
    print(f'Event scoring (same as score_falls.py): merge callback frames with adjacent gap <= {args.gap} gray frames; inclusive interval overlap with GT; MP4 index = gray index * 4. Duplicate matched events do not add FP/FN; repeated callbacks are not separate events.')
    print(f'Accuracy: {output}')


if __name__ == '__main__':
    main()
