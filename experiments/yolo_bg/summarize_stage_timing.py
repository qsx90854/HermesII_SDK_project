#!/usr/bin/env python3
"""Summarize optional SDK timing CSVs from one short run or a batch directory."""
import argparse
import csv
import math
from pathlib import Path


def read(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in row:
            row[k] = float(row[k])
    return rows


def table(records):
    if not records:
        return ['無紀錄。']
    stages = [k for k in records[0][1] if k.endswith('_ms') and k != 'timestamp_ms']
    total = sum(r['total_ms'] for _, r in records)
    lines = ['| 區段 | 有進入的幀數 | 攤每幀 ms | 有進入時平均 ms | P95（全幀）ms | 最大 ms | 最大值位置（index / timestamp）| 佔總時間 |',
             '|---|---:|---:|---:|---:|---:|---|---:|']
    for key in sorted(stages, key=lambda k: sum(r[k] for _, r in records), reverse=True):
        values = sorted(r[key] for _, r in records)
        active = sum(r.get(key[:-3] + '_calls', 1) > 0 for _, r in records)
        name, peak = max(records, key=lambda x: x[1][key])
        summed = sum(values)
        lines.append(f'| {key[:-3]} | {active} | {summed/len(records):.3f} | {summed/active if active else 0:.3f} | {values[math.ceil(.95*len(values))-1]:.3f} | {peak[key]:.3f} | {name}: {int(peak["frame_index"])} / {int(peak["timestamp_ms"])} ms | {summed/total*100 if total else 0:.1f}% |')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    main_files = sorted(args.directory.rglob('algorithm_timing.csv'))
    if not main_files:
        ap.error('找不到 algorithm_timing.csv；請以 HERMES_STAGE_TIMING=1 重新短測，並正常結束。')
    lines = ['# SDK 分段耗時', '', '單位 ms，steady_clock 牆鐘耗時，包含排程等待。各檔內區段互斥，total 不可再加入區段總和；BG 明細已包含於主流程 bg_controller，tracking 明細已包含於主流程 tracking。SDK 外層工作請對照 frames.csv。', '',
             '有進入時平均以影格為單位，不是每個 block 的平均。*_calls 為區段進入次數，包含巢狀區段返回；未執行分支可能只有判斷成本。', '']
    for filename in ['algorithm_timing.csv', 'bg_controller_timing.csv', 'tracking_timing.csv', 'association_detail_timing.csv', 'reconstruction_detail_timing.csv']:
        records, updating, other = [], [], []
        for path in sorted(args.directory.rglob(filename)):
            rows = read(path)
            if not rows:
                continue
            label = str(path.parent.relative_to(args.directory))
            if rows[0]['dropped_frames']:
                lines.append(f'注意：{label}/{filename} 超過容量，另有 {int(rows[0]["dropped_frames"])} 幀未保留；以下僅統計保留部分。')
            events = path.parent / 'person_bg_events.csv'
            stamps = set()
            if events.exists():
                with events.open(newline='') as f:
                    stamps = {int(e['applied_timestamp_ms']) for e in csv.DictReader(f) if e['status'] == 'applied'}
            for row in rows:
                records.append((label, row))
                (updating if int(row['timestamp_ms']) in stamps else other).append((label, row))
            sums = [sum(v for k, v in r.items() if k.endswith('_ms') and k not in ('timestamp_ms', 'total_ms')) for r in rows]
            if any(abs(s-r['total_ms']) > .001 for s, r in zip(sums, rows)):
                raise ValueError(f'{path}: stage sum differs from total')
        lines += [f'## {filename}：全部影格（{len(records)}）', ''] + table(records) + ['']
        lines += [f'### 套用 BG 更新的影格（{len(updating)}）', ''] + table(updating) + ['']
        lines += [f'### 其他影格（{len(other)}）', ''] + table(other) + ['']
        if filename == 'tracking_timing.csv' and records:
            lines += ['### 最慢 10 幀及工作量', '',
                      '| data | index | timestamp ms | tracking ms | 最耗時子區段 | 子區段 ms | 前景像素 | 關聯物件 | tracks | footprint 像素 | cost pairs |',
                      '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
            for label, row in sorted(records, key=lambda x: x[1]['total_ms'], reverse=True)[:10]:
                stage = max((k for k in row if k.endswith('_ms') and k not in ('total_ms', 'timestamp_ms')), key=lambda k: row[k])
                lines.append(f'| {label} | {int(row["frame_index"])} | {int(row["timestamp_ms"])} | {row["total_ms"]:.3f} | {stage[:-3]} | {row[stage]:.3f} | {int(row["fg_pixels"])} | {int(row["association_objects"])} | {int(row["tracks_before"])} | {int(row["footprint_pixels"])} | {int(row["cost_pairs"])} |')
            lines.append('')

    result = '\n'.join(lines)
    if args.output:
        args.output.write_text(result + '\n')
    print(result)


if __name__ == '__main__':
    main()
